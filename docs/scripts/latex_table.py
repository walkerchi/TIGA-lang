"""LaTeX replacement table for docs/scripts/latex_svg.py.

Keys are the exact (whitespace-stripped) text contents of <text> elements in
the docs SVG diagrams; values are LaTeX math sources rendered via matplotlib
mathtext (Computer Modern).  Only genuine formulas are listed — code
identifiers (gf.*, torch.*, method names), shape annotations ((N, F)) and
prose captions intentionally stay as text.
"""

LATEX = {
    # --- attention trio (graph / edge UDF / aggregate panels) ---
    "s = (k[j] · q[i]) / √D":
        r"s_{ij} = \langle k_j, q_i \rangle / \sqrt{D}",
    "α = softmax over j":
        r"\alpha = \mathrm{softmax}_j(s)",
    "α = softmax over j ≤ i":
        r"\alpha = \mathrm{softmax}_{j \leq i}(s)",
    "α = softmax over j ≤ i,":
        r"\alpha = \mathrm{softmax}_{j \leq i}(s),",
    "j in seq(i)":
        r"j \in \mathrm{seq}(i)",
    "out[i] = Σ_j α[i,j] · v[j]":
        r"\mathrm{out}_i = \sum_j \alpha_{ij}\, v_j",
    "out[i] = Σ_{j≤i} α[i,j] · v[j]":
        r"\mathrm{out}_i = \sum_{j \leq i} \alpha_{ij}\, v_j",
    "out[i] = Σ α[i,j] · v[j]":
        r"\mathrm{out}_i = \sum_{j \leq i,\; j \in \mathrm{seq}(i)} \alpha_{ij}\, v_j",
    "α[i,·] ≈ [.62, .21, .11, .06]":
        r"\alpha_{i,\cdot} \approx [0.62,\; 0.21,\; 0.11,\; 0.06]",
    "k₀·v₀": r"k_0,\; v_0",
    "k₁·v₁": r"k_1,\; v_1",
    "k₂·v₂": r"k_2,\; v_2",
    "k₃·v₃": r"k_3,\; v_3",
    "k₄·v₄": r"k_4,\; v_4",
    "qᵢ · dst": r"q_i \; (\mathrm{dst})",
    "q₂ · dst i = 2": r"q_2 \; (\mathrm{dst}\ i{=}2)",
    "q₃ · dst i = 3": r"q_3 \; (\mathrm{dst}\ i{=}3)",

    # --- GCN / diffusion / custom reducer / pagerank ---
    "m[e,:] = w[e] · x[j,:]":
        r"m_{e,:} = w_e\, x_{j,:}",
    "out[i,:] = Σ m[e,:]":
        r"\mathrm{out}_{i,:} = \sum_{e=(j \to i)} m_{e,:}",
    "per edge e = (j→i)":
        r"\mathrm{per\ edge}\ e = (j \to i)",
    "flux = c · (u[j] - u[i])":
        r"\mathrm{flux}_i = c_e\, (u_j - u_i)",
    "u_next = u + dt · flux":
        r"u^{+} = u + \Delta t \cdot \mathrm{flux}",
    "dt = 0.1":
        r"\Delta t = 0.1",
    "v -> (v, 1)":
        r"v \mapsto (v,\, 1)",
    "(s1,n1) ⊕ (s2,n2)":
        r"(s_1, n_1) \oplus (s_2, n_2)",
    "= (s1+s2, n1+n2)":
        r"= (s_1{+}s_2,\; n_1{+}n_2)",
    "(s, n) -> s / n":
        r"(s,\, n) \mapsto s / n",
    "(1.0, 1)": r"(1.0,\, 1)",
    "(4.0, 1)": r"(4.0,\, 1)",
    "(5.0, 2)": r"(5.0,\, 2)",
    "5.0 / 2 = 2.5": r"5.0\, /\, 2 = 2.5",
    "⊕": r"\oplus",
    "rank := step(rank)":
        r"\mathrm{rank} \leftarrow \mathrm{step}(\mathrm{rank})",

    # --- radius / kNN ---
    "m[e] = d[e] · x[j]":
        r"m_e = d_e\, x_j",
    "out[i] = Σ d[e]·x[j]":
        r"\mathrm{out}_i = \sum_{e=(j \to i)} d_e\, x_j",
    "d = 0.3": r"d = 0.3",
    "d = 0.5": r"d = 0.5",
    "p₀ (x=2)": r"p_0 \; (x{=}2)",
    "p₁ (x=3)": r"p_1 \; (x{=}3)",
    "p₂ (x=5)": r"p_2 \; (x{=}5)",
    "m[e] = w[e] · x[j]":
        r"m_e = w_e\, x_j",
    "out[i] = Σ w·x[j]":
        r"\mathrm{out}_i = \sum_{j \in \mathrm{knn}(i,\, k)} w_e\, x_j",

    # --- solvers ---
    "(A u)_i = (2 u_i - u_{i-1} - u_{i+1}) / h = h":
        r"(A u)_i = (2 u_i - u_{i-1} - u_{i+1})\, /\, h = h",
    "u_i": r"u_i",
    "u_{i-1}": r"u_{i-1}",
    "u_{i+1}": r"u_{i+1}",
    "-1/h": r"-1/h",
    "2/h": r"2/h",
    "u=0, x=0": r"u{=}0,\; x{=}0",
    "u=0, x=1": r"u{=}0,\; x{=}1",
    "(A u)_i = m·u_i":
        r"(A u)_i = m\, u_i",
    "+ sum_{j~i} (u_i - u_j)":
        r"+\; \sum_{j \sim i} (u_i - u_j)",
    "= b_i": r"= b_i",
    "p_i": r"p_i",
    "p_j": r"p_j",
    "m = 1, b_i = 1 + (i mod 3)":
        r"m = 1,\; b_i = 1 + (i\ \mathrm{mod}\ 3)",
    "Ap = A · p":
        r"A p",
    "alpha = rho / dot(p, Ap)":
        r"\alpha = \rho\, /\, \langle p, A p \rangle",
    "u += alpha·p;  r -= alpha·Ap":
        r"u \leftarrow u + \alpha p;\quad r \leftarrow r - \alpha A p",
    "p = z + (rho_new / rho) · p":
        r"p \leftarrow z + (\rho_{\mathrm{new}} / \rho)\, p",
    "||r|| ≤ tol":
        r"\| r \| \leq \mathrm{tol}",

    # --- linear recurrence / complex / joint autograd ---
    "k_t ⊗ v_t   (K, V)":
        r"k_t \otimes v_t \;\; (K{,} V)",
    "S_t = S_{t−1} + k_t ⊗ v_t":
        r"S_t = S_{t-1} + k_t \otimes v_t",
    "out_t = q_t · S_t":
        r"\mathrm{out}_t = q_t^{\top}\, S_t",
    "t": r"t",
    "t−1": r"t{-}1",
    "e = conj(y) * y":
        r"e = \overline{y} \odot y",
    "1+2j    3−4j": r"1{+}2j \qquad 3{-}4j",
    "2+0.5j  −1+3j": r"2{+}0.5j \quad\; {-}1{+}3j",
    "[1+2j, 2+0.5j,": r"[1{+}2j,\; 2{+}0.5j,",
    " 3−4j, −1+3j]": r"3{-}4j,\; {-}1{+}3j]",
    "x = [2, 3]": r"x = [2,\, 3]",
    "x² = [4, 9]": r"x^2 = [4,\, 9]",
    "4 + 9": r"4 + 9",
    "loss = 13": r"\mathrm{loss} = 13",

    # --- graph program / torch interop / torch.library ---
    "out0[i] = Σ w0[e]·x[j]":
        r"\mathrm{out}_{0,i} = \sum_{e=(j \to i)} w_{0,e}\, x_j",
    "out1[i] = Σ w1[e]·x[j]":
        r"\mathrm{out}_{1,i} = \sum_{e=(j \to i)} w_{1,e}\, x_j",
    "out[i] = x[(i−1) mod 8] + x[(i+1) mod 8]":
        r"\mathrm{out}_i = x_{(i-1)\ \mathrm{mod}\ 8} + x_{(i+1)\ \mathrm{mod}\ 8}",
    "x[0] = 1": r"x_0 = 1",
    "x[1] = 2": r"x_1 = 2",
    "x[2] = 4": r"x_2 = 4",
    "w = 2": r"w = 2",
    "w = 3": r"w = 3",
    "w = 5": r"w = 5",
    "w = 7": r"w = 7",
    "out[0] = 8": r"\mathrm{out}_0 = 8",
    "out[1] = 38": r"\mathrm{out}_1 = 38",

    # --- edge-NN ---
    "pos[j] − pos[i]":
        r"\mathrm{pos}_j - \mathrm{pos}_i",
    "x[j]": r"x_j",
    "out[i]": r"\mathrm{out}_i",
    "edge (j→i) iff ‖pos[j] − pos[i]‖ ≤ r":
        r"(j \to i) \Leftrightarrow \| \mathrm{pos}_j - \mathrm{pos}_i \| \leq r",
    "i": r"i",

    # --- message-passing flow / reducer anatomy / programming model ---
    "m[e] = w[e]·x[j]":
        r"m_e = w_e\, x_j",
    "a[i] = Σ m[e]":
        r"a_i = \sum_{e=(j \to i)} m_e",
    "out = a + b":
        r"\mathrm{out} = a + b",
    "x[j], b[i], w[e]":
        r"x_j,\; b_i,\; w_e",
    "m₁": r"m_1",
    "m₂": r"m_2",
    "m₃": r"m_3",
    "m₄": r"m_4",
    "s₁ = lift(m₁)": r"s_1 = \mathrm{lift}(m_1)",
    "s₂ = lift(m₂)": r"s_2 = \mathrm{lift}(m_2)",
    "s₃ = lift(m₃)": r"s_3 = \mathrm{lift}(m_3)",
    "s₄ = lift(m₄)": r"s_4 = \mathrm{lift}(m_4)",
    "s₁ ⊕ s₂": r"s_1 \oplus s_2",
    "s₃ ⊕ s₄": r"s_3 \oplus s_4",
    "(s₁⊕s₂)⊕(s₃⊕s₄)": r"(s_1 \oplus s_2) \oplus (s_3 \oplus s_4)",
    "m[e] = c[e] · T[j]":
        r"m_e = c_e\, T_j",
    "a[i] = Σ m[e],  e=(j→i)":
        r"a_i = \sum_{e=(j \to i)} m_e",
    "out[i] = a[i] + b[i]":
        r"\mathrm{out}_i = a_i + b_i",
    "out = [11.1, 8.2, 17.3]":
        r"\mathrm{out} = [11.1,\; 8.2,\; 17.3]",
    "e0 c=2": r"e_0 \; c{=}2",
    "e1 c=3": r"e_1 \; c{=}3",
    "e2 c=4": r"e_2 \; c{=}4",
    "e3 c=5": r"e_3 \; c{=}5",
    "e4 c=6": r"e_4 \; c{=}6",
    "×": r"\times",
    "→": r"\to",

    # --- halo / distributed ---
    "T = max(t_comm, t_int) + t_bnd":
        r"T = \max(t_{\mathrm{comm}},\, t_{\mathrm{int}}) + t_{\mathrm{bnd}}",
    "T = t_comm + t_int + t_bnd":
        r"T = t_{\mathrm{comm}} + t_{\mathrm{int}} + t_{\mathrm{bnd}}",
    "needs x[4]": r"\mathrm{needs}\; x_4",
    "owns x[4]": r"\mathrm{owns}\; x_4",

    # --- tensor matmul ---
    "d lhs = cotangent @ rhsᵀ   (2, 3)":
        r"d\, \mathrm{lhs} = G\, \mathrm{rhs}^{\top} \;\; (2{,} 3)",
    "d rhs = lhsᵀ @ cotangent   (3, 2)":
        r"d\, \mathrm{rhs} = \mathrm{lhs}^{\top}\, G \;\; (3{,} 2)",

    # --- tile-pruned attention ---
    "block_max ≥ m_run − τ":
        r"\mathrm{block\_max} \geq m_{\mathrm{run}} - \tau",

    # --- operator model ---
    "flux = k_face * Δu / d":
        r"\mathrm{flux} = k_{\mathrm{face}}\, \Delta u\, /\, d",
}
