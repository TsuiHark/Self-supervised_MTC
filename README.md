# MTC-MAE
## Title: Self-Supervised Learning Malware Traffic Classification Based on Masked Auto-Encoder
The code corresponds to the paper
## Requirement
pytorch 2.0.1 python 3.8.16
## Dataset
Experiments are conducted on EdgeIIoTset [1], Bot-IoT [2], USTC TFC-2016 [3], CIC-AndMal2017 [4], and ISCX VPN-NonVPN [5] datasets.

[1] M. A. Ferrag, O. Friha, D. Hamouda, L. Maglaras, and H. Janicke, “Edge-IIoTset: A new comprehensive realistic cyber security dataset of IoT and IIoT applications for centralized and federated learning,” IEEE Access, vol. 10, no. 1, pp. 40281–40306, Apr. 2022. (https://ieeexplore.ieee.org/abstract/document/9751703)

[2] N. Koroniotis, N. Moustafa, E. Sitnikova, and B. Turnbull, “Towards the development of realistic botnet dataset in the Internet of Things for network forensic analytics: Bot-iot dataset,” Future Gener. Comput. Syst., vol. 100, no. 1, pp. 779–796, Nov. 2019.

[3] W. Wang, M. Zhu, X. Zeng, X. Ye, and Y. Sheng, “Malware traffic classification using convolutional neural network for representation learning,” in Proc. ICOIN, Da Nang, Vietnam, 2017, pp. 712–717.

[4] A. H. Lashkari, A. F. A. Kadir, L. Taheri, and A. A. Ghorbani, “Toward developing a systematic approach to generate benchmark android malware datasets and classification,” in Proc. ICCST, Montreal, QC, Canada, 2018, pp. 1–7.

[5] G. Draper-Gil, A. H. Lashkari, M. S. I. Mamun, and A. A. Ghorbani, “Characterization of encrypted and vpn traffic using time-related,” in Proc. ICISSP, Rome, Italy, 2016, pp. 407–414.



modules-->basic modules to construct transformer-based model MTC-MAE

models-->proposed model MTCMAE and the comparison methods

utils-->load dataset

main_pretrain--> pre-train on Bigdataset
main_finetune--> finetune on downstream datesets
main_test--> test
main_visulization--> visualization of mask reconstruction
## E-mail
If you have any question, please feel free to contact us by e-mail (1022010403@njupt.edu.cn).
