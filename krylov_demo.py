"""Independent classical Krylov demonstration inspired by IBM Quantum Learning.
Run: python krylov_demo.py   Dependencies: numpy, matplotlib
Source: https://quantum.cloud.ibm.com/learning/en/courses/quantum-diagonalization-algorithms/krylov
The 3x3 validation uses IBM's matrix; the larger toy example is our own.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def krylov_energies(H, start, maximum):
    """Build span{v,Hv,...}; orthogonalize twice for numerical stability."""
    basis, energies = [], []
    q = np.asarray(start, dtype=float)
    q = q / np.linalg.norm(q)
    for _ in range(maximum):
        basis.append(q)
        Q = np.column_stack(basis)
        small_H = Q.T @ H @ Q
        energies.append(np.linalg.eigvalsh((small_H + small_H.T) / 2)[0])
        candidate = H @ q
        for _ in range(2):
            candidate -= Q @ (Q.T @ candidate)
        norm = np.linalg.norm(candidate)
        if norm < 1e-13:
            break
        q = candidate / norm
    return np.array(energies), np.column_stack(basis)


def main():
    out = Path(__file__).resolve().parent
    # Check IBM's worked example: approximate eigenvalues 4, 3, 4-sqrt(2).
    A = np.array([[4., -1., 0.], [-1., 4., -1.], [0., -1., 4.]])
    checks, _ = krylov_energies(A, [1., 0., 0.], 3)
    np.testing.assert_allclose(checks, [4., 3., 4.-np.sqrt(2)], atol=1e-12)
    print('IBM 3x3 check:', checks)
    # Own reproducible toy Hamiltonian: isolated lowest eigenvalue and 199 others.
    # Dimensionless test matrix for learning about Krylov convergence.
    n = 200
    H = np.diag(np.r_[0.25, np.linspace(1., 4., n-1)])
    exact = np.linalg.eigvalsh(H)[0]
    energies, Q = krylov_energies(H, np.ones(n), 20)
    signed_errors = energies - exact
    np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-12)
    assert signed_errors.min() > -1e-12  # Ritz variational bound
    assert np.diff(energies).max() < 1e-12  # nested-space convergence
    errors = np.abs(signed_errors)
    m = np.arange(1, len(energies)+1)
    with (out/'convergence.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subspace_dimension', 'ritz_energy', 'exact_energy', 'absolute_error'])
        w.writerows(zip(m, energies, np.full(len(m), exact), errors))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(8.5,5.3), layout='constrained')
    ax.semilogy(m, errors, '-o', color='#087FAA', lw=2.6, ms=5)
    ax.set_xlabel('Number of Krylov basis vectors')
    ax.set_ylabel(r'Lowest-eigenvalue error $|E_m-E_0|$')
    ax.set_title('Classical Krylov convergence', weight='bold', pad=17)
    ax.set_xticks([1,5,10,15,20])
    ax.grid(axis='y', alpha=.16)
    for side in ('top','right'):
        ax.spines[side].set_visible(False)
    fig.supxlabel('Computed example: 200 × 200 toy Hamiltonian; dimensionless energies', fontsize=10)
    fig.savefig(out/'krylov_convergence.png', dpi=250)
    fig.savefig(out/'krylov_convergence.pdf')
    print(f'Exact eigenvalue: {exact:.8f}; last error: {errors[-1]:.3e}')
    print('Checks passed: orthogonality, variational bound, monotonic convergence.')


if __name__ == '__main__':
    main()
