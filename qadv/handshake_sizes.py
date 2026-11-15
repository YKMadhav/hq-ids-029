"""Measure REAL TLS 1.3 handshake sizes on loopback with OpenSSL 3.5 (ML-KEM, ML-DSA).
A counting TCP proxy sits between s_client and s_server; no tickets, so only the handshake is counted."""
import os, socket, subprocess, threading, time, sys, itertools, pandas as pd, numpy as np
OSSL = os.environ['OSSL']; ENV = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(OSSL) + '/..')
ENV['LD_LIBRARY_PATH'] = os.path.dirname(OSSL).rsplit('/apps', 1)[0]; ENV['OPENSSL_CONF'] = os.path.dirname(OSSL) + '/openssl.cnf'
W = '/tmp/claude-0/-home-claude/593d175b-7bce-50be-9e42-eb7480374129/scratchpad/hs'; os.makedirs(W, exist_ok=True)
def run(*a, **k): return subprocess.run([OSSL, *a], env=ENV, capture_output=True, text=True, **k)

CERTS = {'ECDSA-P256': ['ecparam', '-name', 'prime256v1'], 'RSA-2048': None, 'ML-DSA-44': None, 'ML-DSA-65': None}
def make_cert(name):
    key, crt = f'{W}/{name}.key', f'{W}/{name}.crt'
    if name == 'ECDSA-P256': args = ['req', '-x509', '-newkey', 'ec', '-pkeyopt', 'ec_paramgen_curve:prime256v1']
    elif name == 'RSA-2048': args = ['req', '-x509', '-newkey', 'rsa:2048']
    else: args = ['req', '-x509', '-newkey', name.replace('-', '').lower() if False else name]
    r = run(*args, '-nodes', '-keyout', key, '-out', crt, '-subj', '/CN=localhost', '-days', '2')
    return (key, crt) if r.returncode == 0 else None

def proxy_once(listen_port, target_port, out):
    ls = socket.socket(); ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); ls.bind(('127.0.0.1', listen_port)); ls.listen(1)
    def serve():
        c, _ = ls.accept(); s = socket.create_connection(('127.0.0.1', target_port)); out.update(c2s=0, s2c=0, first_c2s=None, first_s2c=None)
        def pump(src, dst, key, first):
            try:
                while True:
                    d = src.recv(65536)
                    if not d: break
                    if out[first] is None: out[first] = len(d)
                    out[key] += len(d); dst.sendall(d)
            except OSError: pass
            finally:
                try: dst.shutdown(socket.SHUT_WR)
                except OSError: pass
        t1 = threading.Thread(target=pump, args=(c, s, 'c2s', 'first_c2s')); t2 = threading.Thread(target=pump, args=(s, c, 's2c', 'first_s2c'))
        t1.start(); t2.start(); t1.join(); t2.join(); c.close(); s.close(); ls.close()
    t = threading.Thread(target=serve); t.start(); return t

def measure(group, cert, reps=5, base=24000):
    key, crt = cert; rows = []
    for rep in range(reps):
        sp, pp = base + rep * 2, base + rep * 2 + 1
        srv = subprocess.Popen([OSSL, 's_server', '-accept', str(sp), '-cert', crt, '-key', key, '-tls1_3', '-groups', group, '-num_tickets', '0', '-www', '-quiet'],
                               env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.4); out = {}; t = proxy_once(pp, sp, out)
        try:
            cl = subprocess.run([OSSL, 's_client', '-connect', f'127.0.0.1:{pp}', '-tls1_3', '-groups', group], input='Q\n', env=ENV, capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired:
            srv.kill(); srv.wait(); continue
        t.join(timeout=10); srv.kill(); srv.wait()
        ok = 'Verification' in cl.stderr + cl.stdout or 'Cipher is' in cl.stdout
        if out and out.get('c2s'): rows.append({'group': group, 'cert': os.path.basename(crt)[:-4], 'rep': rep, 'ok': ok, **out})
    return rows

if __name__ == '__main__':
    groups = ['X25519', 'secp256r1', 'X25519MLKEM768', 'SecP256r1MLKEM768', 'MLKEM512', 'MLKEM768', 'MLKEM1024']
    certs = {n: make_cert(n) for n in CERTS}; print({k: bool(v) for k, v in certs.items()}, flush=True)
    rows = []
    for cn, ct in certs.items():
        if not ct: continue
        for g in groups:
            r = measure(g, ct, reps=3); rows += r; print(cn, g, r[-1] if r else 'NO DATA', flush=True)
    df = pd.DataFrame(rows); df.to_csv('results/qadv/handshake_sizes.csv', index=False)
    print(df[df.ok].groupby(['cert', 'group'])[['c2s', 's2c', 'first_c2s']].agg(['mean', 'std']).round(1).to_string())
