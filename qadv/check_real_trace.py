"""Compare a real, publicly released X25519MLKEM768 handshake (salrashid123/ml-kem-tls-keyexchange, Go client, Apache-2.0)
with our OpenSSL measurements. Usage: python qadv/check_real_trace.py [pcap]"""
import json, sys
from scapy.all import rdpcap, TCP, Raw
path = sys.argv[1] if len(sys.argv) > 1 else 'data/real_traces/salrashid123_x25519mlkem768_go.cap'
ports = None; c2s = s2c = 0; first = None
for pk in rdpcap(path):
    if TCP in pk and Raw in pk:
        n = len(pk[Raw])
        if ports is None: ports = (pk[TCP].sport, pk[TCP].dport); first = n
        if pk[TCP].sport == ports[0]: c2s += n
        else: s2c += n
out = {'source': 'salrashid123/ml-kem-tls-keyexchange trace.cap (Go client, single handshake)', 'first_client_payload': first, 'client_to_server_bytes': c2s, 'server_to_client_bytes': s2c}
json.dump(out, open('results/qadv/real_trace_check.json', 'w'), indent=1); print(out)
