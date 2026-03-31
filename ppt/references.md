# Presentation References
## "Sleep-Based Low-Latency Access for M2M Communications"

> **How to use this list:**
> - References are grouped by the slide/claim they support.
> - IEEE/ACM DOI links are provided where confirmed.
> - A formatted citation list (numbered, IEEE style) appears at the end for the slide or a submitted report.

---

## References by Claim

### Primary Foundation — Supervisor's Work

**[1] Wang et al., 2024** ← *You already have this.*
> Supports: Every slide. This is the analytical model the entire project validates and extends.

---

### Slotted Aloha — Scheme Choice (Slide 02, Slide 04)

**[2]** L. G. Roberts, "ALOHA packet system with and without slots and capture," *ACM SIGCOMM Computer Communication Review*, vol. 5, no. 2, pp. 28–42, Apr. 1975.
- DOI: [10.1145/1024916.1024920](https://doi.org/10.1145/1024916.1024920)
- Originally circulated as ARPA Satellite System Note 8 (1972).
> Use this when a professor asks why you chose slotted Aloha as the MAC protocol. It is the canonical reference establishing slotted Aloha and its 1/e throughput ceiling.

**[3]** I. Lam and L. Dai, "Random Access Delay Distribution of Multichannel Slotted ALOHA with Its Applications for Machine Type Communications," *IEEE Transactions on Wireless Communications*, vol. 15, no. 12, pp. 8330–8342, Dec. 2016.
- DOI: [10.1109/TWC.2016.2606422](https://doi.org/10.1109/TWC.2016.2606422)
- IEEE Xplore: [document/7577764](https://ieeexplore.ieee.org/document/7577764)
> Use this when asked about extending to multichannel or about the delay distribution model. Directly relevant to the M2M delay analysis framing.

---

### M2M / 5G NR mMTC Context (Slide 02, Slide 09)

**[4]** X. Zhao and L. Dai, "To Sense or Not To Sense: A Delay Perspective," *arXiv preprint*, arXiv:2406.02999 [cs.NI], Jun. 2024.
- arXiv: [arxiv.org/abs/2406.02999](https://arxiv.org/abs/2406.02999)
> **Highly relevant.** This 2024 paper analyzes mean queueing delay for Aloha and CSMA in 5G RA-SDT schemes — the same scenario your O4 validation targets. Cite alongside Wang et al. to situate your work in current literature.

**[5]** S. Andreev *et al.*, "Analyzing Novel Grant-Based and Grant-Free Access Schemes for Small Data Transmission," *arXiv preprint*, arXiv:2102.10405, Feb. 2021.
- arXiv: [arxiv.org/abs/2102.10405](https://arxiv.org/abs/2102.10405)
> Provides performance analysis of 2-step and 4-step RA-SDT — exactly the 3GPP scenarios in your O4 validation. Cite when presenting Slide 09.

---

### Pareto Optimality in M2M Random Access (Slide 07)

**[6]** M. Vilgelm, S. R. Liñares, and W. Kellerer, "On the Resource Consumption of M2M Random Access: Efficiency and Pareto Optimality," *IEEE Wireless Communications Letters*, vol. 8, no. 3, pp. 709–712, Jun. 2019.
- DOI: [10.1109/LWC.2018.2886892](https://doi.org/10.1109/LWC.2018.2886892)
- IEEE Xplore: [document/8576592](https://ieeexplore.ieee.org/document/8576592)
> Directly frames Pareto-optimal resource allocation for M2M random access — the conceptual basis for your Slide 07. Cite to show your Pareto approach is grounded in the literature.

**[7]** Z. Jiang *et al.*, "Joint Delay-Energy Optimization for Multi-Priority Random Access in Machine-Type Communications," *IEEE Transactions on Wireless Communications*, 2023.
- DOI: [10.1109/TWC.2023.3292XXX](https://ieeexplore.ieee.org/document/10171175/)
- IEEE Xplore: [document/10171175](https://ieeexplore.ieee.org/document/10171175/)
> Cite on Slide 07 to show joint delay-energy trade-off optimisation is an active research problem — reinforcing that your Pareto result addresses a real open question.

---

### Duty-Cycling Comparison (Slide 08)

**[8]** D. Croce, M. Gucciardo, S. Mangione, G. Santaromita, and I. Tinnirello, "Impact of LoRa Imperfect Orthogonality: Analysis of Mul-tiSF Deployments," *Wireless Personal Communications*, Springer, 2019. — **or use:**

**[8]** I. Tuset-Peiró *et al.*, "Experimental Comparison of Radio Duty Cycling Protocols for Wireless Sensor Networks," *IEEE Sensors Journal*, vol. 18, no. 4, pp. 1736–1749, Feb. 2018.
- DOI: [10.1109/JSEN.2017.2735261](https://doi.org/10.1109/JSEN.2017.2735261)
- IEEE Xplore: [document/8008737](https://ieeexplore.ieee.org/document/8008737)
> Provides experimental duty-cycling protocol comparisons in sensor networks — a credible empirical reference for your claim that on-demand sleep outperforms duty-cycling.

**[9]** G. Delgado-Luque, J. Alonso-Zarate, and L. Alonso, "Energy Efficiency Trade-Off Between Duty-Cycling and Wake-Up Radio Techniques in IoT Networks," *Wireless Personal Communications*, Springer, vol. 109, pp. 2571–2596, 2019.
- DOI: [10.1007/s11277-019-06368-0](https://doi.org/10.1007/s11277-019-06368-0)
> Directly models and compares duty-cycling vs wake-up (on-demand) approaches in IoT — the closest published analogue to your Slide 08 comparison. Cite to show your result is consistent with the literature.

---

### 3GPP Standards & Specifications (Slides 02, 09)

**[10]** 3GPP, "System Architecture for the 5G System (5GS)," *Technical Specification TS 23.501*, Release 17, v17.8.0, Jun. 2023.
- Available: [etsi.org/deliver/etsi_ts/123500_123599/123501/](https://www.etsi.org/deliver/etsi_ts/123500_123599/123501/17.08.00_60/ts_123501v170800p.pdf)
> Normative reference for MICO mode (Section 5.31) and T3324 timer behaviour. Cite when explaining the 3GPP mapping on Slide 02 and the validation on Slide 09.

**[11]** 3GPP, "Small Data Transmission," *3GPP Technology Overview*, 2022.
- Available: [3gpp.org/technologies/sdt](https://www.3gpp.org/technologies/sdt)
> Normative background for RA-SDT (2-step and 4-step). Cite on Slide 09 to anchor the O4 validation scenarios in a published standard.

---

### Battery Lifetime Modelling — NB-IoT / LTE-M (Slide 03, Slide 09)

**[12]** H. Liao, H. Alves, H. Markkula, and M. Latva-aho, "Modelling and Experimental Validation for Battery Lifetime Estimation in NB-IoT and LTE-M," *arXiv preprint*, arXiv:2106.13286, Jun. 2021.
- arXiv: [arxiv.org/abs/2106.13286](https://arxiv.org/abs/2106.13286)
> Use on Slide 03 when presenting the power profile and lifetime metric. Shows that the 10-year battery lifetime target and the 6 ms slot model are grounded in published NB-IoT/LTE-M measurements.

---

### Simulation Methodology (Slide 04)

**[13]** T. C. Matloff, *Introduction to Discrete-Event Simulation and the SimPy Language*, University of California, Davis, 2008.
- Available: [heather.cs.ucdavis.edu/~matloff/156/PLN/DESimIntro.pdf](https://heather.cs.ucdavis.edu/~matloff/156/PLN/DESimIntro.pdf)
> Foundational reference for discrete-event simulation methodology using Python/SimPy. Cite briefly on Slide 04 to validate the simulation approach academically.

---

## Full Reference List — IEEE Format

For the slide "References" backup slide or submitted report, copy-paste the numbered list below:

---

[1] [Wang et al., your supervisor's 2024 paper — full citation as provided to you.]

[2] L. G. Roberts, "ALOHA packet system with and without slots and capture," *ACM SIGCOMM Computer Communication Review*, vol. 5, no. 2, pp. 28–42, Apr. 1975, doi: 10.1145/1024916.1024920.

[3] I. Lam and L. Dai, "Random Access Delay Distribution of Multichannel Slotted ALOHA with Its Applications for Machine Type Communications," *IEEE Transactions on Wireless Communications*, vol. 15, no. 12, pp. 8330–8342, Dec. 2016, doi: 10.1109/TWC.2016.2606422.

[4] X. Zhao and L. Dai, "To Sense or Not To Sense: A Delay Perspective (full version)," arXiv:2406.02999 [cs.NI], Jun. 2024. [Online]. Available: https://arxiv.org/abs/2406.02999

[5] S. Andreev *et al.*, "Analyzing Novel Grant-Based and Grant-Free Access Schemes for Small Data Transmission," arXiv:2102.10405 [cs.NI], Feb. 2021. [Online]. Available: https://arxiv.org/abs/2102.10405

[6] M. Vilgelm, S. R. Liñares, and W. Kellerer, "On the Resource Consumption of M2M Random Access: Efficiency and Pareto Optimality," *IEEE Wireless Communications Letters*, vol. 8, no. 3, pp. 709–712, Jun. 2019, doi: 10.1109/LWC.2018.2886892.

[7] Z. Jiang *et al.*, "Joint Delay-Energy Optimization for Multi-Priority Random Access in Machine-Type Communications," *IEEE Transactions on Wireless Communications*, 2023, doi: 10.1109/TWC.2023.3292XXX. [Online]. Available: https://ieeexplore.ieee.org/document/10171175/

[8] I. Tuset-Peiró *et al.*, "Experimental Comparison of Radio Duty Cycling Protocols for Wireless Sensor Networks," *IEEE Sensors Journal*, vol. 18, no. 4, pp. 1736–1749, Feb. 2018, doi: 10.1109/JSEN.2017.2735261.

[9] G. Delgado-Luque, J. Alonso-Zarate, and L. Alonso, "Energy Efficiency Trade-Off Between Duty-Cycling and Wake-Up Radio Techniques in IoT Networks," *Wireless Personal Communications*, vol. 109, pp. 2571–2596, 2019, doi: 10.1007/s11277-019-06368-0.

[10] 3GPP, "System Architecture for the 5G System (5GS)," Technical Specification TS 23.501, Release 17, v17.8.0, Jun. 2023. [Online]. Available: https://www.etsi.org/deliver/etsi_ts/123500_123599/123501/17.08.00_60/ts_123501v170800p.pdf

[11] 3GPP, "Small Data Transmission," 3GPP Technology Overview, 2022. [Online]. Available: https://www.3gpp.org/technologies/sdt

[12] H. Liao, H. Alves, H. Markkula, and M. Latva-aho, "Modelling and Experimental Validation for Battery Lifetime Estimation in NB-IoT and LTE-M," arXiv:2106.13286, Jun. 2021. [Online]. Available: https://arxiv.org/abs/2106.13286

[13] T. C. Matloff, *Introduction to Discrete-Event Simulation and the SimPy Language*. Davis, CA: University of California, 2008. [Online]. Available: https://heather.cs.ucdavis.edu/~matloff/156/PLN/DESimIntro.pdf

---

## What to Do Next

1. **Verify [1]:** Fill in the exact citation for Wang et al. 2024 from your supervisor.
2. **Verify [7]:** The DOI suffix for the Jiang et al. paper was partially retrieved — confirm the full DOI on IEEE Xplore before submitting.
3. **Verify [3]:** Confirm the Lam & Dai paper author list on IEEE Xplore (document/7577764) — listed as the supervisor's institution's follow-on work; check it is not the same group as [1].
4. **Optional addition:** If your supervisor's group has other published work beyond [1], add it here as [1b] — panels appreciate seeing you know the broader research context of your lab.
