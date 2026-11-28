"""Freeze a separate physical byte-stream transport study on ten old development images."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from test_cheng_byte_transport import verify
r=Path(__file__).resolve().parent;run=r/'runs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
contract=verify();assert contract['passed']==8 and contract['failed']==0
contract.update(source_sha256={n:sha(r/n) for n in ('cheng_byte_transport.py','test_cheng_byte_transport.py','packet_arq_accounting.py')})
(run/'stage66_transport_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
grid=read('stage66_rate_grid.json');audit=read('stage66_rate_grid_audit.json');assert audit['all_checks_passed'] and audit['complete']
assert audit['source_result_sha256']==sha(run/'stage66_rate_grid.json')
selected=audit['selection']['selected_arm_id'];assert selected
arm=next(v for v in grid['arms'] if v['arm_id']==selected)
streams=[{k:v[k] for k in ('image_index','image_sha256','container_path','container_sha256')} for v in arm['calibration_cases']]
for v in streams:assert sha(r/v['container_path'])==v['container_sha256']
plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Independent clean-codec-container physical transport on ten development images, not paper baseline quality reproduction',
    selected_arm_id=selected,checkpoint_sha256=arm['checkpoint_sha256'],grid_audit_sha256=sha(run/'stage66_rate_grid_audit.json'),streams=streams,
    trials_per_image=5,snr_db=[6.,6.5,7.,7.5,8.,9.],expected_frames=300,maximum_attempts=32,
    pilot_coordinates=[[0,0],[0,1],[9,0],[9,1]],pilot_expected_frames=24,noise_seed_base=2026112804,
    noise='PCG64(SeedSequence([base,image,trial,snr_index,packet,attempt])); complex float32 AWGN variance=10**(-snr/10)',
    framing='Contiguous bytes, zero pad last 5-byte packet, 40 payload bits + CRC11, conv [101,111] unterminated 102 bits +2 zeros ->26 16QAM symbols',
    stopping_rules='single pass; first valid receiver CRC (including undetected wrong acceptances); first exact sender payload (ideal oracle only)',
    corruption_accounting='Compare final reassembled container bytes only AFTER each stopping rule; never use truth to extend receiver CRC ARQ. Wrong/censored receptions remain in results.',
    decoder_policy='Do not call native entropy decoder on corrupted streams. This study measures byte delivery and resources only; no noisy-image quality or channel-codec performance claim.',
    diagnostic_noiseless_images=[0,9],tensorflow_threads=2,
    contract_sha256=sha(run/'stage66_transport_contract.json'),source_sha256={n:sha(r/n) for n in ('cache_stage66_transport.py','audit_stage66_transport.py','cheng_byte_transport.py','packet_arq_accounting.py','smoke_stage25_sionna_link.py')},
    prior_frozen_artifacts=read('stage66_rate_grid_plan.json')['historical_evidence_sha256'],
    summary_gate='Only complete 50-frame SNR groups receive rate/delivery means; otherwise coverage and exact aggregate counts only',
    custom_header_transmitted=True,shape_and_checkpoint_metadata_assumptions='CH65 shape header is transmitted; checkpoint identity and original container size shared externally by the frozen plan',
    feedback_airtime_included=False,cmi_airtime_included=False,channel_header_identified=False,paper_baseline_ready=False,full_paper_reproduction_complete=False)
out=run/'stage66_transport_plan.json';assert not out.exists();out.write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps(dict(plan_sha256=sha(out),expected_frames=300,pilot_frames=24,contract_checks=8)))
