#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import sys, os
import librosa
import soundfile as sf
import time
import pickle as pic

import torch
import torch.optim as optim
import torch.nn.functional as F

from Base import EPS, MIC_INDEX, Base, MultiSTFT


class ARMA_JD(Base):
    """
    X_FTM: the observed complex spectrogram
    Q_FMM: diagonalizer that converts a spatial covariance matrix (SCM) to a diagonal matrix
    G_NM: diagonal elements of the diagonalized SCMs
    W_NsFK: basis vectors for each source
    H_NKTd: activations for each source
    lambda_NFTd: power spectral densities of each source (W_NsFK @ H_NKTd)
    Px_power_FTM: power spectra of Qx
    Y_FTM: \sum_n lambda_NFTd G_NM
    """

    method_name = "ARMA_JD"

    def __init__(
        self,
        n_speech=2,
        n_noise=0,
        speech_model=["NMF", "FreqInv", "DNN"][0],
        noise_model=["TimeInv", "NMF"][0],
        n_basis=8,
        xp=np,
        mode=["IP", "ISS1", "ISS2"][0],
        init_SCM="circular",
        speech_VAE=None,
        n_tap_AR=3,
        n_tap_MA=2,
        n_delay_AR=3,
        n_tap_direct=0,
        n_bit=64,
        g_eps=1e-2,
        lr=1e-3,
    ):
        """initialize ARMA_JD

        Parameters:
        -----------
            n_speech: int
                The number of sources
            n_noise: int
                The number of noise.
            n_iter: int
                The number of iteration to update all variables
            n_basis: int
                The number of bases of each source
            init_SCM: str
                How to initialize covariance matrix {unit, obs, ILRMA}
            n_tap_MA: int
                Frame length for early reflection
            n_tap_AR: int
                Frame length for late reverberation
            n_delay: int
                The index to indicate the beginning of the late reverberation ( > n_tap_MA )
        """
        super(ARMA_JD, self).__init__(xp=xp, n_bit=n_bit)
        self.n_speech = n_speech
        self.n_noise = n_noise
        self.n_src = self.n_speech + self.n_noise
        self.speech_model = speech_model
        self.noise_model = noise_model if self.n_noise > 0 else None
        self.mode = mode

        self.n_basis = n_basis
        self.init_SCM = init_SCM
        self.n_tap_MA = n_tap_MA  # Ld
        self.n_tap_AR = n_tap_AR  # Lr
        self.n_delay_AR = n_delay_AR  # Delta_r
        self.n_tap_direct = n_tap_direct
        self.g_eps = g_eps
        self.lr = lr

        if self.n_tap_MA == 0:
            self.n_tap_direct = 0
        if self.n_tap_AR == 0:
            self.n_delay_AR = 0

        self.speech_VAE = speech_VAE
        # assert self.speech_model == "DNN" and self.speech_VAE is not None

        self.method_name = "ARMA_JD"

    def __str__(self):
        src_model_name = f"NMF_K={self.n_basis}" if "NMF" == self.speech_model else self.speech_model
        noise_model_name = f"NMF_K={self.n_basis}" if "NMF" == self.noise_model else self.noise_model

        filename_suffix = (
            f"M={self.n_mic}-S={self.n_speech}-N={self.n_noise}-F={self.n_freq}-it={self.n_iter}"
            f"-src={src_model_name}-noise={noise_model_name}-init={self.init_SCM}"
            f"-LMA={self.n_tap_MA}-LAR={self.n_tap_AR}-DAR={self.n_delay_AR}"
        )
        if self.n_bit == 32:
            filename_suffix += "-bit=32"
        if hasattr(self, "file_id"):
            filename_suffix += f"-ID={self.file_id}"
        return filename_suffix

    def calculate_log_likelihood(self):
        self.calculate_PSD()
        self.calculate_Px()
        self.calculate_Y()
        self.calculate_Px_power()
        self.log_likelihood = (
            -(self.Px_power_FTM / self.Y_FTM + self.xp.log(self.Y_FTM)).sum()
            + self.n_time
            * (self.xp.log(self.xp.linalg.det(self.Q_FMM @ self.Q_FMM.transpose(0, 2, 1).conj()).real)).sum()
        )
        return self.log_likelihood

    def check_likelihood(func):
        def _check_likelihood(self, *args, **kwargs):
            func(self)
            self.reset_variable()
            self.calculate_log_likelihood()
            if hasattr(self, "prev_log_likelihood"):
                if self.log_likelihood + 0.5 < self.prev_log_likelihood:
                    print(
                        f"\nNG {self.it} after updating {func} : diff {self.prev_log_likelihood-self.log_likelihood} now {self.log_likelihood}"
                    )
            self.prev_log_likelihood = self.log_likelihood

        return _check_likelihood

    def load_spectrogram(self, X_FTM, sample_rate=16000):
        super().load_spectrogram(X_FTM)

        self.Xbar_FxTxMLt = self.xp.zeros(
            [self.n_freq, self.n_time, self.n_mic * (self.n_tap_AR + 1)], dtype=self.TYPE_COMPLEX
        )
        self.Xbar_FxTxMLt[:, :, : self.n_mic] = self.X_FTM
        for l in range(self.n_tap_AR):
            self.Xbar_FxTxMLt[:, self.n_delay_AR + l :, (l + 1) * self.n_mic : (l + 2) * self.n_mic] = self.X_FTM[
                :, : -(self.n_delay_AR + l)
            ]
        self.Px_FTM = self.X_FTM.copy()

    def init_source_model(self):
        if self.speech_model == "NMF":
            self.W_NsFK = self.xp.random.rand(self.n_speech, self.n_freq, self.n_basis).astype(self.TYPE_FLOAT)
            self.H_NsKT = self.xp.random.rand(self.n_speech, self.n_basis, self.n_time).astype(self.TYPE_FLOAT)
        elif self.speech_model == "FreqInv":
            self.lambda_NsT = self.xp.random.rand(self.n_speech, self.n_time).astype(self.TYPE_FLOAT) + EPS
        elif self.speech_model == "DNN":
            self.U_NsF = self.xp.ones([self.n_speech, self.n_freq])
            self.V_NsT = self.xp.ones([self.n_speech, self.n_time])
            self.torch_device = "cpu" if self.xp is np else f"cuda:{self.X_FTM.device.id}"

        if self.n_noise > 0:
            if self.noise_model == "NMF":
                self.W_noise_NnFK = self.xp.random.rand(self.n_noise, self.n_freq, self.n_basis).astype(
                    self.TYPE_FLOAT
                )
                self.H_noise_NnKT = self.xp.random.rand(self.n_noise, self.n_basis, self.n_time).astype(
                    self.TYPE_FLOAT
                )
            elif self.noise_model == "TimeInv":
                self.W_noise_NnFK = self.xp.ones([self.n_noise, self.n_freq, 1]).astype(self.TYPE_FLOAT)
                self.H_noise_NnKT = self.xp.ones([self.n_noise, 1, self.n_time]).astype(self.TYPE_FLOAT)
        self.lambda_NFT = self.xp.zeros([self.n_src, self.n_freq, self.n_time], dtype=self.TYPE_FLOAT)

        self.calculate_PSD()
        self.P_FxMxMLt = self.xp.zeros(
            [self.n_freq, self.n_mic, self.n_mic * (self.n_tap_AR + 1)], dtype=self.TYPE_COMPLEX
        )
        self.Px_FTM = self.X_FTM.copy()

    def init_spatial_model(self):
        self.start_idx = 0
        self.Q_FMM = self.xp.tile(self.xp.eye(self.n_mic), [self.n_freq, 1, 1]).astype(self.TYPE_COMPLEX)
        self.P_FxMxMLt[:, :, : self.n_mic] = self.Q_FMM
        self.G_NLdM = self.xp.maximum(
            self.g_eps, self.xp.zeros([self.n_speech, self.n_tap_MA + 1, self.n_mic], dtype=self.TYPE_FLOAT)
        )

        if "circular" in self.init_SCM:
            for m in range(self.n_mic):
                self.G_NLdM[m % self.n_speech, 0, m] = 1

        elif "twostep" in self.init_SCM:
            self.start_idx = 50

            separater_init = ARMA_JD(
                n_speech=self.n_speech,
                n_noise=self.n_noise,
                speech_model="FreqInv",
                noise_model="TimeInv",
                init_SCM="circular",
                xp=self.xp,
                n_bit=self.n_bit,
                n_tap_MA=0,
                n_tap_AR=self.n_tap_AR,
                n_delay_AR=self.n_delay_AR,
                g_eps=self.g_eps,
            )
            separater_init.file_id = self.file_id
            separater_init.load_spectrogram(self.X_FTM)
            separater_init.solve(n_iter=self.start_idx, save_wav=False)
            self.P_FxMxMLt = separater_init.P_FxMxMLt
            self.Q_FMM = self.P_FxMxMLt[:, :, : self.n_mic]

            self.G_NLdM = self.g_eps * self.xp.ones(
                [self.n_speech, self.n_tap_MA + 1, self.n_mic], dtype=self.TYPE_FLOAT
            )
            self.G_NLdM[:, 0] = separater_init.G_NLdM[:, 0]

            if self.speech_model == "DNN":
                power_speech_NsxFxT = self.xp.asarray(
                    self.xp.abs(separater_init.separated_spec[: self.n_speech]) ** 2
                ).astype(self.xp.float32)
                power_speech_NsxFxT /= power_speech_NsxFxT.sum(axis=1).mean(axis=1)[:, None, None]
                with torch.set_grad_enabled(False):
                    self.Z_NsDT = self.speech_VAE.encode_(
                        torch.as_tensor(power_speech_NsxFxT + EPS, device=self.torch_device)
                    ).detach()
                    self.Z_NsDT.requires_grad = True
                    self.z_optimizer = optim.AdamW([self.Z_NsDT], lr=self.lr)
                    self.power_speech_NsxFxT = self.xp.asarray(self.speech_VAE.decode_(self.Z_NsDT))

        else:
            print(f"Please specify how to initialize covariance matrix {separater.init_SCM}")
            raise ValueError

        self.P_FxMxMLt[:, :, self.n_mic :] = 0
        self.G_NLdM /= self.G_NLdM.sum(axis=(1, 2))[:, None, None]
        self.normalize()

    def reset_variable(self):
        self.calculate_Px()
        self.calculate_Px_power()
        self.calculate_Y()

    def calculate_PSD(self):
        if self.speech_model == "NMF":
            self.lambda_NFT[: self.n_speech] = self.W_NsFK @ self.H_NsKT + EPS
        elif self.speech_model == "FreqInv":
            self.lambda_NFT[: self.n_speech] = self.lambda_NsT[:, None]
        elif self.speech_model == "DNN":
            self.lambda_NFT[: self.n_speech] = self.U_NsF[:, :, None] * self.V_NsT[:, None] * self.power_speech_NsxFxT

        if self.n_noise > 0:
            self.lambda_NFT[self.n_speech :] = self.W_noise_NnFK @ self.H_noise_NnKT

    def calculate_Px(self):
        self.Px_FTM = (self.P_FxMxMLt[:, None] * self.Xbar_FxTxMLt[:, :, None]).sum(axis=3)

    def calculate_Px_power(self):
        self.Px_power_FTM = self.xp.abs(self.Px_FTM) ** 2

    def calculate_Yn(self):
        self.calculate_PSD()
        self.Y_NFTM = self.lambda_NFT[:, :, :, None] * self.G_NLdM[:, 0, None, None]
        if self.n_tap_direct > 0:
            for l in range(1, self.n_tap_direct + 1):
                self.Y_NFTM[:, :, l:] += self.lambda_NFT[:, :, :-l, None] * self.G_NLdM[:, l, None, None]

    def calculate_Y(self):
        self.calculate_PSD()
        self.Y_FTM = (self.lambda_NFT[:, :, :, None] * self.G_NLdM[:, 0, None, None]).sum(axis=0)
        for l in range(1, 1 + self.n_tap_MA):
            self.Y_FTM[:, l:] += (self.lambda_NFT[:, :, :-l, None] * self.G_NLdM[:, l, None, None]).sum(axis=0)
        self.Y_FTM += EPS

    def update(self):
        self.update_PSD()
        self.update_G()
        self.update_AR()
        self.normalize()

    def update_PSD(self):
        if self.speech_model == "NMF":
            self.update_PSD_NMF()
        elif self.speech_model == "FreqInv":
            self.update_PSD_FreqInv()
        elif self.speech_model == "DNN":
            self.update_PSD_DNN()

        if self.noise_model == "NMF":
            self.update_PSD_NMF_noise()

    def update_PSD_NMF(self):
        XY2_FTM = self.Px_power_FTM / (self.Y_FTM**2)
        HG_NsKLTM = self.H_NsKT[:, :, None, :, None] * self.G_NLdM[: self.n_speech, None, :, None]
        HG_sum_NsKTM = HG_NsKLTM[:, :, 0].copy()  # copyいらなさそう
        for l in range(1, 1 + self.n_tap_MA):
            HG_sum_NsKTM[:, :, l:] += HG_NsKLTM[:, :, l, :-l]
        a_W_NsFK = (HG_sum_NsKTM[:, None] * XY2_FTM[None, :, None]).sum(axis=(3, 4))
        b_W_NsFK = (HG_sum_NsKTM[:, None] / self.Y_FTM[None, :, None]).sum(axis=(3, 4))

        self.W_NsFK *= self.xp.sqrt(a_W_NsFK / b_W_NsFK)
        self.calculate_Y()

        XY2_FTM = self.Px_power_FTM / (self.Y_FTM**2)
        WXY2_NsKTM = (self.W_NsFK[:, :, :, None, None] * XY2_FTM[None, :, None]).sum(axis=1)
        WYinv_NKTM = (self.W_NsFK[:, :, :, None, None] / self.Y_FTM[None, :, None]).sum(axis=1)
        GWXY2_NsKTM = self.G_NLdM[: self.n_speech, 0, None, None] * WXY2_NsKTM
        GWYinv_NsKTM = self.G_NLdM[: self.n_speech, 0, None, None] * WYinv_NKTM
        for l in range(1, 1 + self.n_tap_MA):
            GWXY2_NsKTM[:, :, :-l] += self.G_NLdM[: self.n_speech, l, None, None] * WXY2_NsKTM[:, :, l:]
            GWYinv_NsKTM[:, :, :-l] += self.G_NLdM[: self.n_speech, l, None, None] * WYinv_NKTM[:, :, l:]
        a_H_NsKT = GWXY2_NsKTM.sum(axis=3)
        b_H_NsKT = GWYinv_NsKTM.sum(axis=3)

        self.H_NsKT *= self.xp.sqrt(a_H_NsKT / b_H_NsKT)
        self.calculate_Y()

    def update_PSD_NMF_noise(self):
        XY2_FTM = self.Px_power_FTM / (self.Y_FTM**2)
        HG_NnKLTM = self.H_noise_NnKT[:, :, None, :, None] * self.G_NLdM[self.n_speech :, None, :, None]
        HG_sum_NnKTM = HG_NnKLTM[:, :, 0].copy()  # copyいらなさそう
        for l in range(1, 1 + self.n_tap_MA):
            HG_sum_NnKTM[:, :, l:] += HG_NnKLTM[:, :, l, :-l]
        a_W_NnFK = (HG_sum_NnKTM[:, None] * XY2_FTM[None, :, None]).sum(axis=(3, 4))
        b_W_NnFK = (HG_sum_NnKTM[:, None] / self.Y_FTM[None, :, None]).sum(axis=(3, 4))

        self.W_noise_NnFK *= self.xp.sqrt(a_W_NnFK / b_W_NnFK)
        self.calculate_Y()

        XY2_FTM = self.Px_power_FTM / (self.Y_FTM**2)
        WXY2_NnKTM = (self.W_noise_NnFK[:, :, :, None, None] * XY2_FTM[None, :, None]).sum(axis=1)
        WYinv_NKTM = (self.W_noise_NnFK[:, :, :, None, None] / self.Y_FTM[None, :, None]).sum(axis=1)
        GWXY2_NnKTM = self.G_NLdM[self.n_speech :, 0, None, None] * WXY2_NnKTM
        GWYinv_NnKTM = self.G_NLdM[self.n_speech :, 0, None, None] * WYinv_NKTM
        for l in range(1, 1 + self.n_tap_MA):
            GWXY2_NnKTM[:, :, :-l] += self.G_NLdM[self.n_speech :, l, None, None] * WXY2_NnKTM[:, :, l:]
            GWYinv_NnKTM[:, :, :-l] += self.G_NLdM[self.n_speech :, l, None, None] * WYinv_NKTM[:, :, l:]
        a_H_NnKT = GWXY2_NnKTM.sum(axis=3)
        b_H_NnKT = GWYinv_NnKTM.sum(axis=3)

        self.H_noise_NnKT *= self.xp.sqrt(a_H_NnKT / b_H_NnKT)
        self.calculate_Y()

    def update_PSD_FreqInv(self):
        XY2_TM = (self.Px_power_FTM / (self.Y_FTM**2)).sum(axis=0)
        GXY2_NsLTM = self.G_NLdM[: self.n_speech, :, None] * XY2_TM[None, None]
        GY_NsLTM = self.G_NLdM[: self.n_speech, :, None] * (1 / self.Y_FTM).sum(axis=0)[None, None]
        GXY2_sum_NsTM = GXY2_NsLTM[:, 0].copy()
        GY_sum_NsTM = GY_NsLTM[:, 0].copy()
        for l in range(1, 1 + self.n_tap_MA):
            GXY2_sum_NsTM[:, :-l] += GXY2_NsLTM[:, l, l:]
            GY_sum_NsTM[:, :-l] += GY_NsLTM[:, l, l:]
        self.lambda_NsT *= self.xp.sqrt(GXY2_sum_NsTM.sum(axis=2) / GY_sum_NsTM.sum(axis=2))
        self.calculate_Y()

    def update_PSD_DNN(self):
        XY2_FTM = self.Px_power_FTM / (self.Y_FTM**2)
        VZG_NsFTLM = (self.V_NsT[:, :, None, None] * self.G_NLdM[: self.n_speech, None])[
            :, None
        ] * self.power_speech_NsxFxT[..., None, None]
        VZG_sum_NsFTM = VZG_NsFTLM[:, :, :, 0]
        for l in range(1, 1 + self.n_tap_MA):
            VZG_sum_NsFTM[:, :, l:] += VZG_NsFTLM[:, :, :-l, l]
        a_U_NsF = (VZG_sum_NsFTM * XY2_FTM[None]).sum(axis=(2, 3))
        b_U_NsF = (VZG_sum_NsFTM / self.Y_FTM[None]).sum(axis=(2, 3))

        self.U_NsF *= self.xp.sqrt(a_U_NsF / b_U_NsF)
        self.calculate_Y()

        XY2_FTM = self.Px_power_FTM / (self.Y_FTM**2)
        GXY2_NFTL = (self.G_NLdM[: self.n_speech, None, None] * XY2_FTM[None, :, :, None]).sum(axis=-1)
        GYinv_NFTL = (self.G_NLdM[: self.n_speech, None, None] / self.Y_FTM[None, :, :, None]).sum(axis=-1)
        GXY2_sum_NFT = GXY2_NFTL[..., 0].copy()
        GYinv_sum_NFT = GYinv_NFTL[..., 0].copy()
        for l in range(1, 1 + self.n_tap_MA):
            GXY2_sum_NFT[:, :, :-l] += GXY2_NFTL[:, :, l:, l]
            GYinv_sum_NFT[:, :, :-l] += GYinv_NFTL[:, :, l:, l]
        a_V_NsT = ((self.U_NsF[:, :, None] * self.power_speech_NsxFxT) * GXY2_sum_NFT).sum(axis=1)
        b_V_NsT = ((self.U_NsF[:, :, None] * self.power_speech_NsxFxT) * GYinv_sum_NFT).sum(axis=1)

        self.V_NsT *= self.xp.sqrt(a_V_NsT / b_V_NsT)
        self.calculate_Y()

    def loss_fn(self, Y_noise_FTM_torch, G_NLdM_torch, UV_NsFT_torch):  # for update Z by backprop
        power_speech_NsxFxT = self.speech_VAE.decode_(self.Z_NsDT)
        lambda_tmp_NsFT = UV_NsFT_torch * power_speech_NsxFxT  # + EPS
        if self.n_tap_MA > 0:
            lambda_tmp_NsFT = F.pad(lambda_tmp_NsFT, [self.n_tap_MA, 0], mode="constant", value=0)
            Y_tmp_FTM = (
                (lambda_tmp_NsFT[:, :, self.n_tap_MA :, None] * G_NLdM_torch[: self.n_speech, 0, None, None]).sum(
                    axis=0
                )
                + Y_noise_FTM_torch
                + EPS
            )
            for l in range(1, 1 + self.n_tap_MA):
                Y_tmp_FTM = Y_tmp_FTM + (
                    lambda_tmp_NsFT[:, :, self.n_tap_MA - l : -l, None] * G_NLdM_torch[: self.n_speech, l, None, None]
                ).sum(axis=0)
        else:
            Y_tmp_FTM = (
                (lambda_tmp_NsFT[..., None] * G_NLdM_torch[: self.n_speech, 0, None, None]).sum(axis=0)
                + Y_noise_FTM_torch
                + EPS
            )
        return (
            torch.log(Y_tmp_FTM) + torch.as_tensor(self.Px_power_FTM, device=self.torch_device) / Y_tmp_FTM
        ).sum() / (self.n_freq * self.n_mic)

    def update_Z(self):
        if self.n_noise > 0:
            Y_noise_FTM_torch = (
                self.lambda_NFT[self.n_speech :, :, :, None] * self.G_NLdM[self.n_speech :, 0, None, None]
            ).sum(axis=0)
            for l in range(1, 1 + self.n_tap_MA):
                Y_noise_FTM_torch[:, l:] += (
                    self.lambda_NFT[self.n_speech :, :, :-l, None] * self.G_NLdM[self.n_speech :, l, None, None]
                ).sum(axis=0)
        else:
            Y_noise_FTM_torch = self.xp.zeros_like(self.X_FTM, dtype=self.TYPE_FLOAT)
        Y_noise_FTM_torch = torch.as_tensor(Y_noise_FTM_torch, device=self.torch_device)
        G_NLdM_torch = torch.as_tensor(self.G_NLdM, device=self.torch_device)
        UV_NsFT_torch = torch.as_tensor(self.U_NsF[:, :, None] * self.V_NsT[:, None], device=self.torch_device)

        for it in range(self.n_Z_iteration):
            self.z_optimizer.zero_grad()
            loss = self.loss_fn(Y_noise_FTM_torch, G_NLdM_torch, UV_NsFT_torch)
            loss.backward()
            self.z_optimizer.step()

        with torch.set_grad_enabled(False):
            self.power_speech_NsxFxT = self.xp.asarray(self.speech_VAE.decode_(self.Z_NsDT))

    # @check_likelihood
    # @check_nan
    def update_G(self):
        XY2_FTM = self.Px_power_FTM / (self.Y_FTM**2)
        a_G_NM = ((self.lambda_NFT)[..., None] * XY2_FTM[None]).sum(axis=(1, 2))
        b_G_NM = ((self.lambda_NFT)[..., None] / self.Y_FTM[None]).sum(axis=(1, 2))
        self.G_NLdM[:, 0] *= self.xp.sqrt(a_G_NM / b_G_NM)
        for l in range(1, 1 + self.n_tap_MA):
            a_G_NM = ((self.lambda_NFT[:, :, :-l])[..., None] * XY2_FTM[None, :, l:]).sum(axis=(1, 2))
            b_G_NM = ((self.lambda_NFT[:, :, :-l])[..., None] / self.Y_FTM[None, :, l:]).sum(axis=(1, 2))
            self.G_NLdM[:, l] *= self.xp.sqrt(a_G_NM / b_G_NM)
        self.calculate_Y()

    # @check_likelihood
    def update_AR(self):
        if self.mode == "IP":
            for m in range(self.n_mic):
                Vinv_FxMLtxMLt = self.xp.linalg.inv(
                    self.xp.einsum(
                        "fti, ftj -> fij", self.Xbar_FxTxMLt, self.Xbar_FxTxMLt.conj() / self.Y_FTM[:, :, m, None]
                    )
                    / self.n_time
                )
                u_FM = self.xp.linalg.inv(self.P_FxMxMLt[:, :, : self.n_mic])[:, :, m]
                self.P_FxMxMLt[:, m] = (
                    (Vinv_FxMLtxMLt[:, :, : self.n_mic] * u_FM[:, None]).sum(axis=2)
                    / self.xp.sqrt(
                        (u_FM.conj() * (Vinv_FxMLtxMLt[:, : self.n_mic, : self.n_mic] * u_FM[:, None]).sum(axis=2))
                        .sum(axis=1)
                        .real
                    )[:, None]
                ).conj()
            self.calculate_Px()

        elif "ISS" in self.mode:
            for m in range(self.n_mic):
                QdQd_FTM = self.Px_FTM * self.Px_FTM[:, :, m, None].conj()
                V_tmp_FxM = (QdQd_FTM[:, :, m, None] / self.Y_FTM).mean(axis=1)
                V_FxM = (QdQd_FTM / self.Y_FTM).mean(axis=1) / V_tmp_FxM
                V_FxM[:, m] = 1 - 1 / self.xp.sqrt(V_tmp_FxM[:, m])
                self.Px_FTM -= self.xp.einsum("fm, ft -> ftm", V_FxM, self.Px_FTM[:, :, m])
                self.P_FxMxMLt -= self.xp.einsum("fi, fj -> fij", V_FxM, self.P_FxMxMLt[:, m])

            if self.n_tap_AR > 0:
                if self.mode == "ISS1":
                    for m in range(self.n_mic, (self.n_tap_AR + 1) * self.n_mic):
                        a_FxM = ((self.Px_FTM / self.Y_FTM) * self.Xbar_FxTxMLt[:, :, m, None].conj()).sum(axis=1)
                        b_FxM = ((self.xp.abs(self.Xbar_FxTxMLt[:, :, m]) ** 2)[:, :, None] / self.Y_FTM).sum(axis=1)
                        V_FxM = a_FxM / b_FxM
                        self.P_FxMxMLt[:, :, m] -= V_FxM
                        self.Px_FTM -= V_FxM[:, None] * self.Xbar_FxTxMLt[:, :, m, None]
                elif self.mode == "ISS2":
                    a_FxMxML = self.xp.einsum(
                        "ftm, fti -> fmi", self.Px_FTM / self.Y_FTM, self.Xbar_FxTxMLt[:, :, self.n_mic :].conj()
                    )
                    c_FxMxML = self.xp.zeros(
                        [self.n_freq, self.n_mic, self.n_mic * self.n_tap_AR], dtype=self.TYPE_COMPLEX
                    )
                    for m in range(self.n_mic):
                        b_FxMLxML = self.xp.linalg.inv(
                            self.xp.einsum(
                                "fti, ftj -> fij",
                                self.Xbar_FxTxMLt[:, :, self.n_mic :] / self.Y_FTM[:, :, m, None],
                                self.Xbar_FxTxMLt[:, :, self.n_mic :].conj(),
                            )
                        )
                        c_FxMxML[:, m] = self.xp.einsum("fi, fij -> fj", a_FxMxML[:, m], b_FxMLxML)
                    self.P_FxMxMLt[:, :, self.n_mic :] -= c_FxMxML
                    self.Px_FTM -= (c_FxMxML[:, None] @ self.Xbar_FxTxMLt[:, :, self.n_mic :, None]).squeeze()

        self.Q_FMM = self.P_FxMxMLt[:, :, : self.n_mic]
        self.calculate_Px_power()

    # @check_likelihood
    def normalize(self):
        if self.speech_model in ["NMF", "DNN"]:
            phi_F = self.xp.sum(self.Q_FMM * self.Q_FMM.conj(), axis=(1, 2)).real / self.n_mic
            self.P_FxMxMLt /= self.xp.sqrt(phi_F)[:, None, None]
            if self.speech_model == "NMF":
                self.W_NsFK /= phi_F[None, :, None]
            elif self.speech_model == "DNN":
                self.U_NsF /= phi_F[None]
            if self.n_noise > 0:
                self.W_noise_NnFK /= phi_F[None, :, None]

        mu_N = (self.G_NLdM).sum(axis=(1, 2))
        self.G_NLdM /= mu_N[:, None, None]
        if self.speech_model == "NMF":
            self.W_NsFK *= mu_N[: self.n_speech, None, None]
        elif self.speech_model == "FreqInv":
            self.lambda_NsT *= mu_N[: self.n_speech, None]
        elif self.speech_model == "DNN":
            self.U_NsF *= mu_N[: self.n_speech, None]
        if self.n_noise > 0:
            self.W_noise_NnFK *= mu_N[self.n_speech :, None, None]

        if self.speech_model == "NMF":
            nu_NsK = self.W_NsFK.sum(axis=1)
            self.W_NsFK /= nu_NsK[:, None]
            self.H_NsKT *= nu_NsK[:, :, None]

        if self.speech_model == "DNN":
            nu_Ns = self.U_NsF.sum(axis=1)
            self.U_NsF /= nu_Ns[:, None]
            self.V_NsT *= nu_Ns[:, None]

        if self.n_noise > 0:
            nu_NnK = self.W_noise_NnFK.sum(axis=1)
            self.W_noise_NnFK /= nu_NnK[:, None]
            self.H_noise_NnKT *= nu_NnK[:, :, None]

        self.reset_variable()

    def separate(self, mic_index=MIC_INDEX):
        self.calculate_Yn()
        self._separate(mic_index=mic_index)

    def separate_direct(self, mic_index=MIC_INDEX):
        self.calculate_Yn_direct()
        self._separate(mic_index=mic_index)

    def _separate(self, mic_index=MIC_INDEX):
        self.calculate_Y()
        self.calculate_Px()
        Q_inv_FMM = self.xp.linalg.inv(self.Q_FMM)

        for n in range(self.n_speech):
            tmp = (Q_inv_FMM[:, None, mic_index] * (self.Y_NFTM[n] / self.Y_FTM * self.Px_FTM)).sum(axis=2)
            if n == 0:
                self.separated_spec = np.zeros([self.n_speech, self.n_freq, self.n_time], dtype=np.complex128)
            self.separated_spec[n] = self.convert_to_NumpyArray(tmp)
        return self.separated_spec

    def save_param(self, filename="test.pic"):
        param_list = [self.W_NsFK, self.H_NsKT, self.G_NLdM, self.P_FxMxMLt]
        if self.xp != np:
            param_list = [self.convert_to_NumpyArray(param) for param in param_list]
        pic.dump(param_list, open(filename, "wb"))

    def load_param(self, filename):
        param_list = pic.load(open(filename, "rb"))
        if self.xp != np:
            param_list = [self.xp.asarray(param) for param in param_list]
        self.W_NsFK, self.H_NsKT, self.G_NLdM, self.P_FxMxMLt = param_list

        self.n_speech, self.n_freq, self.n_basis = self.W_NsFK.shape
        self.n_time = self.H_NsKT[2]
        self.n_tap_MA = self.G_NLdM[1] - 1
        self.n_tap_AR = (self.P_FxMxMLt.shape[2] / self.n_mic) - 1


if __name__ == "__main__":
    import argparse
    import pickle as pic
    import sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("input_fname", type=str, help="filename of the multichannel observed signals")
    parser.add_argument("--file_id", type=str, default="None", help="file id")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--n_fft", type=int, default=1024, help="number of frequencies")
    parser.add_argument("--n_speech", type=int, default=3, help="number of speech")
    parser.add_argument("--n_noise", type=int, default=0, help="number of noise")
    parser.add_argument("--n_basis", type=int, default=16, help="number of basis")
    parser.add_argument("--n_tap_MA", type=int, default=8, help="filter length for MA model")
    parser.add_argument("--n_tap_AR", type=int, default=4, help="filter length for AR model")
    parser.add_argument("--n_delay_AR", type=int, default=3, help="delay parameter for AR model")
    parser.add_argument("--init_SCM", type=str, default="twostep", help="circular or twostep")
    parser.add_argument("--n_iter", type=int, default=100, help="number of iteration")
    parser.add_argument("--n_mic", type=int, default=8, help="number of microphone")
    parser.add_argument("--n_bit", type=int, default=64, help="number of microphone")
    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp

        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()

    wav, sample_rate = sf.read(args.input_fname)
    wav /= np.abs(wav).max() * 1.2
    M = min(len(wav), args.n_mic)
    spec_FTM = MultiSTFT(wav[:, :M], n_fft=args.n_fft)

    separater = ARMA_JD(
        n_speech=args.n_speech,
        n_noise=args.n_noise,
        speech_model=["NMF", "FreqInv", "DNN"][0],
        noise_model=["TimeInv", "NMF"][0],
        n_basis=args.n_basis,
        mode=["IP", "ISS1", "ISS2"][0],
        speech_VAE=None,
        n_tap_direct=0,
        g_eps=1e-2,
        lr=1e-3,
        xp=xp,
        init_SCM=args.init_SCM,
        n_tap_MA=args.n_tap_MA,
        n_tap_AR=args.n_tap_AR,
        n_delay_AR=args.n_delay_AR,
        n_bit=args.n_bit,
    )
    separater.file_id = args.file_id
    separater.load_spectrogram(spec_FTM, sample_rate)
    separater.n_iter = args.n_iter
    separater.solve(
        n_iter=args.n_iter,
        save_likelihood=False,
        save_param=False,
        save_wav=False,
        save_dir="./",
        interval_save=100,
    )
