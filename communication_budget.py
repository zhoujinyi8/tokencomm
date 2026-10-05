"""Communication-use accounting for the TokCom and TaiChi reproduction paths.

The two paths solve different tasks, so this module keeps source representation
rate, coded digital-link use, and analog semantic-channel use as separate units.
It deliberately does not convert continuous semantic symbols into bits without
an explicit quantizer and entropy-coding model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def tokcom_budget(
    *,
    num_tokens: int,
    codebook_size: int,
    image_height: int,
    image_width: int,
    packet_tokens: int = 4,
    crc_bits: int = 16,
    termination_bits: int = 6,
    coded_bits_per_input_bit: int = 2,
    modulation_bits_per_symbol: int = 4,
) -> dict:
    """Return exact framing use for this project's terminated digital link.

    Packet headers, interleaver descriptions, synchronization, pilots, channel
    estimation, and retransmissions are not represented by ``digital_link.py``
    and are therefore excluded here as well.
    """
    num_tokens = _positive_int("num_tokens", num_tokens)
    codebook_size = _positive_int("codebook_size", codebook_size)
    image_height = _positive_int("image_height", image_height)
    image_width = _positive_int("image_width", image_width)
    packet_tokens = _positive_int("packet_tokens", packet_tokens)
    coded_bits_per_input_bit = _positive_int("coded_bits_per_input_bit", coded_bits_per_input_bit)
    modulation_bits_per_symbol = _positive_int("modulation_bits_per_symbol", modulation_bits_per_symbol)
    if codebook_size < 2 or codebook_size & (codebook_size - 1):
        raise ValueError("codebook_size must be a power of two")
    if crc_bits < 0 or termination_bits < 0:
        raise ValueError("crc_bits and termination_bits must be nonnegative")

    token_bits = int(math.log2(codebook_size))
    packet_payload_tokens = [
        min(packet_tokens, num_tokens - start) for start in range(0, num_tokens, packet_tokens)
    ]
    payload_bits_by_packet = [count * token_bits for count in packet_payload_tokens]
    coded_bits_by_packet = [
        (payload + crc_bits + termination_bits) * coded_bits_per_input_bit
        for payload in payload_bits_by_packet
    ]
    symbols_by_packet = [
        math.ceil(coded / modulation_bits_per_symbol) for coded in coded_bits_by_packet
    ]
    padding_by_packet = [
        symbols * modulation_bits_per_symbol - coded
        for symbols, coded in zip(symbols_by_packet, coded_bits_by_packet)
    ]
    pixels = image_height * image_width
    source_bits = num_tokens * token_bits
    coded_bits = sum(coded_bits_by_packet)
    symbols = sum(symbols_by_packet)
    return {
        "path": "tokcom_digital",
        "task_unit": "one image",
        "num_tokens": num_tokens,
        "codebook_size": codebook_size,
        "token_bits": token_bits,
        "source_bits": source_bits,
        "source_bits_per_pixel": source_bits / pixels,
        "packet_tokens": packet_tokens,
        "packets": len(packet_payload_tokens),
        "crc_bits_per_packet": crc_bits,
        "termination_bits_per_packet": termination_bits,
        "coded_bits_per_input_bit": coded_bits_per_input_bit,
        "coded_bits": coded_bits,
        "qam_padding_bits": sum(padding_by_packet),
        "modulation_bits_per_symbol": modulation_bits_per_symbol,
        "complex_channel_symbols": symbols,
        "complex_symbols_per_pixel": symbols / pixels,
        "coded_bits_per_source_bit": coded_bits / source_bits,
        "source_bits_per_complex_symbol": source_bits / symbols,
        "image_pixels": pixels,
        "excluded_overhead": [
            "packet headers",
            "interleaver signaling",
            "synchronization",
            "pilots and channel estimation",
            "retransmissions",
        ],
    }


def taichi_budget(
    *,
    visual_tokens: int,
    real_channel_dimension: int,
    image_height: int,
    image_width: int,
) -> dict:
    """Return semantic-channel use without claiming an undefined bit rate."""
    visual_tokens = _positive_int("visual_tokens", visual_tokens)
    real_channel_dimension = _positive_int("real_channel_dimension", real_channel_dimension)
    image_height = _positive_int("image_height", image_height)
    image_width = _positive_int("image_width", image_width)
    if real_channel_dimension % 2:
        raise ValueError("real_channel_dimension must be even because Channel pairs I/Q values")
    pixels = image_height * image_width
    real_values = visual_tokens * real_channel_dimension
    symbols = real_values // 2
    return {
        "path": "taichi_analog_semantic",
        "task_unit": "one image/question inference",
        "visual_tokens": visual_tokens,
        "real_channel_dimension_per_token": real_channel_dimension,
        "real_channel_values": real_values,
        "complex_channel_symbols": symbols,
        "complex_symbols_per_pixel": symbols / pixels,
        "image_pixels": pixels,
        "bit_rate": None,
        "bit_rate_reason": (
            "The learned channel values are continuous. A bit count requires a separately specified "
            "quantizer, entropy code, framing scheme, and digital modulation/FEC stack."
        ),
        "excluded_overhead": [
            "question/text delivery",
            "synchronization",
            "pilots and channel estimation",
            "model delivery and computation",
        ],
    }


def default_report() -> dict:
    paper_scale = tokcom_budget(
        num_tokens=256, codebook_size=1024, image_height=256, image_width=256, packet_tokens=4
    )
    tiny = tokcom_budget(
        num_tokens=64, codebook_size=64, image_height=32, image_width=32, packet_tokens=4
    )
    taichi = taichi_budget(
        visual_tokens=16, real_channel_dimension=64, image_height=224, image_width=224
    )
    return {
        "unit_policy": {
            "source_bpp": "discrete source-token bits divided by source-image pixels",
            "complex_channel_symbols": "complex modulation symbols or complex analog channel uses",
            "no_cross_task_efficiency_claim": (
                "TokCom reconstructs an image while the current TaiChi run answers an Imagenette "
                "classification question; raw symbol counts alone do not establish superiority."
            ),
        },
        "configurations": {
            "tokcom_public_vqgan_reference_link": paper_scale,
            "tokcom_tiny_baseline": tiny,
            "taichi_stage10_stage11": taichi,
        },
        "descriptive_symbol_ratio": {
            "tokcom_reference_over_taichi": (
                paper_scale["complex_channel_symbols"] / taichi["complex_channel_symbols"]
            ),
            "interpretation": "descriptive only because image reconstruction and classification/VQA are different tasks",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    report = default_report()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
