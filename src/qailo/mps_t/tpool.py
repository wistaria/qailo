import numpy as np

from ..mps_p.projector import _full_projector


class tpool:
    def __init__(self):
        self.tpool = []
        self.gpool = []

    def _register(self, tensor):
        id = len(self.tpool)
        self.tpool.append([tensor, "initial", None])
        return id

    def _contract(self, tid0, ss0, tid1, ss1, ss2=None):
        id = len(self.tpool)
        if ss2 is None:
            t = np.einsum(self.tpool[tid0][0], ss0, self.tpool[tid1][0], ss1)
        else:
            t = np.einsum(self.tpool[tid0][0], ss0, self.tpool[tid1][0], ss1, ss2)
        self.tpool.append([t, "product", tid0, ss0, tid1, ss1, ss2])
        return id

    def _trim(self, T, d):
        """trim last dimension of tensor"""
        assert d <= T.shape[-1]
        shape = T.shape
        T = T.reshape((np.prod(shape[:-1]), shape[-1]))
        return T[:, :d].reshape(shape[:-1] + (d,))

    def _projector(self, t0, ss0, t1, ss1, maxdim=None):
        S, d, WLh, WR = _full_projector(t0, ss0, t1, ss1)
        d = d if maxdim is None else min(d, maxdim)
        gid = len(self.gpool)
        self.gpool.append([S, d, WLh, WR])
        lid = len(self.tpool)
        self.tpool.append([self._trim(WLh, d), "squeezer L", gid])
        rid = len(self.tpool)
        self.tpool.append([self._trim(WR, d), "squeezer R", gid])
        return gid, lid, rid

    def _dump(self, prefix):
        import json

        dic = {"prefix": prefix}
        tlist = []
        for id, tp in enumerate(self.tpool):
            m = {}
            m["id"] = id
            m["type"] = tp[1]
            m["shape"] = tp[0].shape
            if tp[1] == "product":
                m["from 0"] = tp[2]
                m["subscripts 0"] = tp[3]
                m["from 1"] = tp[4]
                m["subscripts 1"] = tp[5]
                if tp[6] is not None:
                    m["subscripts"] = tp[6]
            elif tp[1] == "squeezer L" or tp[1] == "squeezer R":
                m["from"] = tp[2]
            tlist.append(m)
            if tp[1] == "initial":
                np.save(f"{prefix}-tensor-{id}", tp[0])
        dic["tensor"] = tlist
        glist = []
        for id, gp in enumerate(self.gpool):
            m = {}
            m["id"] = id
            m["dim from"] = gp[0].shape[0]
            m["dim to"] = gp[1]
            m["shape L"] = gp[2].shape
            m["shape R"] = gp[3].shape
            glist.append(m)
            np.savez(f"{prefix}-generator-{id}", s=gp[0], wl=gp[1], wr=gp[2])
        dic["generator"] = glist

        with open(prefix + "-graph.json", mode="w") as f:
            json.dump(dic, f, indent=2)

    def _load(self, prefix):
        import json

        with open(prefix + "-graph.json", mode="r") as f:
            dic = json.load(f)
            print(dic)
            assert "prefix" in dic and dic["prefix"] == prefix

            self.tpool = []
            if "tensor" in dic:
                for t in dic["tensor"]:
                    id = t["id"]
                    if t["type"] == "initial":
                        self.tpool.append(
                            [np.load(f"{prefix}-tensor-{id}.npy"), "initial", None]
                        )
                        assert list(self.tpool[-1][0].shape) == t["shape"]
                    elif t["type"] == "product":
                        if "subscripts" not in dic:
                            t["subscripts"] = None
                        self.tpool.append(
                            [
                                None,
                                "product",
                                t["from 0"],
                                t["subscripts 0"],
                                t["from 1"],
                                t["subscripts 1"],
                                t["subscripts"],
                            ]
                        )
                    elif "squeezer L" in dic or "squeezer R" in dic:
                        self.tpool.append([None, t["type"], t["from"]])
            self.gpool = []
            if "generator" in dic:
                for g in dic["generator"]:
                    id = g["id"]
                    data = np.load(f"{prefix}-generator-{id}.npz")
                    self.gpool.append([data["s"], g["dim to"], data["wl"], data["wr"]])
