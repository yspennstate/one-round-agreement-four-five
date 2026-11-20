"""Every formula and number the paper quotes, generated here (n=4 from verify_exact/regcheck, n=5 from n5regimes)."""
import sympy as sp
import n5regimes as N, verify_exact as X
p = N.p; q = 1 - p
print('--- n=5 switch polynomials (full) and roots')
S = N.build()
for (n1, a, b, lam, opt), nxt in zip(S, S[1:]):
    r = b; print(n1, '->', nxt[0], float(r), sp.Poly(sp.minimal_polynomial(r, p), p).as_expr())
print('--- n=5 OPT per regime: factored numerator/denominator')
for name, a, b, lam, opt in S:
    nu, de = sp.fraction(sp.cancel(opt))
    print(name, 'OPT =', sp.factor(nu), '/', sp.factor(de))
print('--- n=5 lambda=0 regimes against the AMP formula')
amp2 = 1 - ((1 - q**2)**3 + (1 - q**3)**2) / 2
print('AMP regime OPT == 1-1/2[(1-q^2)^3+(1-q^3)^2]:', sp.expand(S[1][4] - amp2) == 0)
print('low regime OPT expanded in q:', sp.factor(sp.expand(S[0][4].subs(p, 1 - sp.Symbol('q')))))
print('--- n=5 exact values at 1/2 and lambda')
for name, a, b, lam, opt in S:
    if float(a) <= 0.5 <= float(b): print(name, 'OPT(1/2) =', sp.nsimplify(opt.subs(p, sp.Rational(1, 2))), 'lambda =', sp.nsimplify(sp.sympify(lam).subs(p, sp.Rational(1, 2))))
print('--- table OPT_4, OPT_5, q at p = 0.1..0.9')
def opt4(t):
    t = sp.Rational(t)
    if t <= 1 - 2**sp.Rational(-1, 4): return 1 - 2*(1-t)**4*(1-(1-t)**2)**2
    if t <= sp.Rational(4245, 10000): return 1 - (1 - (1-t)**2)**2
    if t <= sp.Rational(1, 2): return X.opt_one(t)
    return X.opt_half(t)
def opt5(t):
    for name, a, b, lam, opt in S:
        if float(a) <= t <= float(b): return float(opt.subs(p, sp.Rational(str(t)))), name
for t in ['0.05', '0.1', '0.2', '0.3', '0.4', '0.45', '0.5', '0.6', '0.7', '0.8', '0.9']:
    v5, nm = opt5(float(t))
    print(t, '%.6f' % float(opt4(t)), '%.6f %s' % (v5, nm), '%.6f' % (1 - float(t)))
