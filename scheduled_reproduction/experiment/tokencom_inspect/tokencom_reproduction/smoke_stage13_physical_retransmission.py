"""Small development-only physical retransmission integration check.

Receiver uses CRC decisions and decoded payloads only. Feedback packet IDs are
assumed delivered reliably; feedback airtime and headers are not simulated.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

from digital_link import bits_int, int_bits, transmit_packet
from pretrained_vq import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--dependency-dir', type=Path)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path,
                   default=Path('data/imagenette_stage5/token_cache/val.pt'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--images', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261029)
    p.add_argument('--snr', type=float, default=8.)
    p.add_argument('--budget', type=int, default=4)
    p.add_argument('--random-trials', type=int, default=1)
    p.add_argument('--feedback-attempts', type=int, default=0,
                   help='0 assumes reliable feedback; positive values send coded feedback with retries')
    p.add_argument('--feedback-seed', type=int, default=20261029)
    p.add_argument('--feedback-header-bits', type=int, choices=(0, 16), default=0)
    args = p.parse_args()
    if args.out.exists(): p.error('Output already exists')
    if args.random_trials < 1: p.error('Random trials must be positive')
    if args.feedback_attempts < 0: p.error('Feedback attempts cannot be negative')
    if args.feedback_attempts and args.budget != 4:
        p.error('Coded feedback currently requires exactly four packet IDs')
    if args.dependency_dir: sys.path.insert(0, str(args.dependency_dir.resolve()))
    sys.path.insert(0, str(args.reference.resolve()))
    from maskgit.configs import maskgit_class_cond_config
    from maskgit.nets.bidirectional_transformer import BidirectionalTransformer

    torch.set_num_threads(4)
    data = torch.load(args.data, map_location='cpu', weights_only=True)
    cfg = maskgit_class_cond_config.get_config()
    model = BidirectionalTransformer(
        num_image_tokens=257, num_codebook_vectors=2025,
        dim=cfg.transformer.num_embeds, n_layers=cfg.transformer.num_layers,
        hidden_dim=cfg.transformer.intermediate_size,
        num_heads=cfg.transformer.num_heads,
        attention_dropout=cfg.transformer.dropout_rate,
        hidden_dropout=cfg.transformer.dropout_rate).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu',
                                     weights_only=True)['state_dict'], strict=True)

    def predict(received, missing):
        seq = torch.cat((torch.tensor([2024]), received.masked_fill(missing, 2024)))[None]
        with torch.inference_mode():
            logits = model(seq)[:, 1:, :1024][0]
            conf, pred = logits.softmax(-1).max(-1)
        return conf, torch.where(missing, pred, received)

    rows = []
    for image_index in range(min(args.images, len(data['ids']))):
        target = data['ids'][image_index].numpy().astype(np.int64)
        rng = np.random.default_rng(args.seed + image_index)
        order = rng.permutation(256)
        received = np.zeros(256, dtype=np.int64)
        missing = np.zeros(256, dtype=bool)
        initial_ok = np.zeros(64, dtype=bool)
        for packet in range(64):
            pos = order[packet * 4:(packet + 1) * 4]
            payload = int_bits(target[pos], 10).reshape(-1)
            decoded, ok, _ = transmit_packet(payload, args.snr, rng, 'rayleigh')
            initial_ok[packet] = ok
            if ok: received[pos] = bits_int(decoded.reshape(-1, 10))
            else: missing[pos] = True
        original_missing = missing.copy()
        initial_received = received.copy()
        initial_missing = missing.copy()
        initial_rng_state = copy.deepcopy(rng.bit_generator.state)
        feedback_rng = np.random.default_rng(args.feedback_seed + args.seed * 1000 + image_index)
        feedback_rng_state = copy.deepcopy(feedback_rng.bit_generator.state)
        conf, before = predict(torch.from_numpy(received), torch.from_numpy(missing))
        conf = conf.numpy()
        lost_packets = np.flatnonzero(~initial_ok).tolist()
        chosen_low = sorted(lost_packets,
                            key=lambda k: (float(conf[order[k * 4:(k + 1) * 4]].mean()), k))[:args.budget]
        def retransmit(selected):
            sent_ids = list(selected)
            feedback_attempts_used = 0
            feedback_crc_passed = True
            if args.feedback_attempts:
                feedback_crc_passed = False
                control_rng = np.random.default_rng()
                control_rng.bit_generator.state = copy.deepcopy(feedback_rng_state)
                ids_bits = int_bits(np.asarray(selected, dtype=np.int64), 6).reshape(-1)
                header = (int_bits((1 << 12) | image_index, 16).reshape(-1)
                          if args.feedback_header_bits else np.empty(0, dtype=np.uint8))
                bits = np.concatenate((header, ids_bits))
                if len(ids_bits) != 24: raise ValueError('Coded feedback requires exactly four packet IDs')
                for feedback_attempts_used in range(1, args.feedback_attempts + 1):
                    decoded_ids, ok, _ = transmit_packet(bits, args.snr, control_rng, 'rayleigh')
                    if ok:
                        if args.feedback_header_bits and not np.array_equal(decoded_ids[:16], header):
                            raise RuntimeError('CRC-undetected feedback header error')
                        sent_ids = bits_int(decoded_ids[args.feedback_header_bits:].reshape(4, 6)).tolist()
                        feedback_crc_passed = True
                        break
                if not feedback_crc_passed: sent_ids = []
            local_rng = np.random.default_rng()
            local_rng.bit_generator.state = copy.deepcopy(initial_rng_state)
            rec = initial_received.copy()
            miss = initial_missing.copy()
            successes = 0
            for packet in sent_ids:
                pos = order[packet * 4:(packet + 1) * 4]
                payload = int_bits(target[pos], 10).reshape(-1)
                decoded, ok, _ = transmit_packet(payload, args.snr, local_rng, 'rayleigh')
                successes += int(ok)
                if ok:
                    rec[pos] = bits_int(decoded.reshape(-1, 10))
                    miss[pos] = False
            _, recovered = predict(torch.from_numpy(rec), torch.from_numpy(miss))
            correct = int((recovered.numpy()[original_missing] == target[original_missing]).sum())
            result = {'selected_packet_ids': selected, 'crc_passed_retransmissions': successes,
                    'originally_missing_token_hits': correct,
                    'remaining_missing_tokens': int(miss.sum())}
            if args.feedback_attempts:
                result.update(decoded_feedback_packet_ids=sent_ids,
                              feedback_crc_passed=feedback_crc_passed,
                              feedback_attempts_used=feedback_attempts_used)
            return result

        random_trials = []
        for trial in range(args.random_trials):
            chosen_random = np.random.default_rng(
                args.seed + 100000 + image_index + trial * 1000000).choice(
                    lost_packets, size=min(args.budget, len(lost_packets)), replace=False).tolist()
            random_trials.append(retransmit(chosen_random))
        rows.append({'source_index': image_index,
                     'initial_crc_failed_packets': len(lost_packets),
                     'initial_missing_tokens': int(original_missing.sum()),
                     'initial_originally_missing_token_hits': int(
                         (before.numpy()[original_missing] == target[original_missing]).sum()),
                     'low_confidence': retransmit(chosen_low),
                     'random': random_trials[0],
                     'random_trials': random_trials})
        print(json.dumps(rows[-1]), flush=True)
    result = {'scope': 'small development-only physical link smoke check',
              'channel': 'rayleigh', 'snr_definition': 'Es/N0', 'snr_db': args.snr,
              'feedback': 'reliable packet IDs assumed; overhead not counted',
              'seed': args.seed, 'budget_packets': args.budget,
              'random_trials_per_image': args.random_trials,
              'checkpoint_sha256': sha256(args.checkpoint),
              'data_identity': data['identity'], 'rows': rows}
    if args.feedback_attempts:
        result['feedback'] = 'coded packet IDs, CRC-gated, independent Rayleigh channel, timeout retry assumed'
        result['feedback_attempts'] = args.feedback_attempts
        result['feedback_seed'] = args.feedback_seed
        result['feedback_header_bits'] = args.feedback_header_bits
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
