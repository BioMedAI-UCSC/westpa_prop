import io

import numpy as np

from embeddings.base_embedding import BaseEmbedding


class CVAEEmbedding(BaseEmbedding):
    """Convolutional VAE on inter-chain contact maps (Paper 2 style).

    Maps are resized to (img_size, img_size) so any system works. transform()
    returns the latent mean. Configurable latent dim, channels, training.
    """

    def __init__(self, n_components=2, img_size=64, channels=(16, 32, 64),
                 epochs=40, lr=1e-3, batch=128, beta=1.0, device=None, seed=0):
        super().__init__(n_components)
        self.img_size = int(img_size)
        self.channels = tuple(channels)
        self.epochs, self.lr, self.batch = int(epochs), float(lr), int(batch)
        self.beta, self.seed = float(beta), int(seed)
        self.device = device
        self.net = None

    # -- torch model -------------------------------------------------------- #
    def _build(self):
        import torch.nn as nn
        S, ch = self.img_size, self.channels
        enc, c = [], 1
        for nc in ch:
            enc += [nn.Conv2d(c, nc, 3, 2, 1), nn.ReLU()]
            c = nc
        red = S // (2 ** len(ch))
        flat = c * red * red

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = nn.Sequential(*enc)
                s.fc_mu = nn.Linear(flat, self.n_components)
                s.fc_lv = nn.Linear(flat, self.n_components)
                s.fc_d = nn.Linear(self.n_components, flat)
                dec, cc = [], c
                for nc in list(reversed(ch[:-1])) + [1]:
                    dec += [nn.ConvTranspose2d(cc, nc, 4, 2, 1),
                            nn.ReLU() if nc != 1 else nn.Sigmoid()]
                    cc = nc
                s.dec = nn.Sequential(*dec)
                s.red, s.c = red, c

            def encode(s, x):
                h = s.enc(x).flatten(1)
                return s.fc_mu(h), s.fc_lv(h)

            def decode(s, z):
                h = s.fc_d(z).view(-1, s.c, s.red, s.red)
                return s.dec(h)

            def forward(s, x):
                mu, lv = s.encode(x)
                std = (0.5 * lv).exp()
                import torch
                z = mu + std * torch.randn_like(std)
                return s.decode(z), mu, lv

        return Net()

    def _to_img(self, X):
        import torch
        import torch.nn.functional as F
        X = torch.as_tensor(np.asarray(X, np.float32))
        if X.ndim == 3:
            X = X.unsqueeze(1)
        return F.interpolate(X, size=(self.img_size, self.img_size),
                             mode="bilinear", align_corners=False)

    def _dev(self):
        return self.safe_device(self.device)

    def fit(self, X, lengths=None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(self.seed)
        dev = self._dev()
        self.net = self._build().to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(self._to_img(X)),
                            batch_size=self.batch, shuffle=True)
        self.net.train()
        for _ in range(self.epochs):
            for (xb,) in loader:
                xb = xb.to(dev)
                recon, mu, lv = self.net(xb)
                rl = torch.nn.functional.binary_cross_entropy(recon, xb, reduction="sum")
                kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
                loss = (rl + self.beta * kl) / xb.size(0)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def transform(self, X):
        import torch
        dev = self._dev()
        self.net.eval()
        out = []
        with torch.no_grad():
            imgs = self._to_img(X).to(dev)
            for i in range(0, imgs.size(0), 1024):
                mu, _ = self.net.encode(imgs[i:i + 1024])
                out.append(mu.cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def save(self, path):
        import torch
        torch.save({"cfg": {k: getattr(self, k) for k in
                            ("n_components", "img_size", "channels", "epochs",
                             "lr", "batch", "beta", "seed")},
                    "state": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path):
        import torch
        d = torch.load(path, map_location="cpu")
        obj = cls(**d["cfg"])
        obj.net = obj._build()
        obj.net.load_state_dict(d["state"])
        return obj
