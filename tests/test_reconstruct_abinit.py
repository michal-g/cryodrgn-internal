"""Running ab-initio volume reconstruction followed by downstream analyses."""

import pytest
import argparse
import os.path
from cryodrgn.commands import analyze, abinit


@pytest.mark.parametrize(
    "particles, indices, ctf",
    [
        ("hand", None, "CTF-Test.100"),
        ("toy.txt", "random-100", "CTF-Test"),
        ("toy.star", "first-100", "CTF-Test"),
    ],
    indirect=True,
    ids=["hand,no.ind", "toy.txt,ind.rand.100", "toy.star,ind.f100"],
)
class TestAbinitHetero:
    def get_outdir(self, tmpdir_factory, particles, ctf, indices):
        dirname = os.path.join("AbinitHet", particles.label, ctf.label, indices.label)
        odir = os.path.join(tmpdir_factory.getbasetemp(), dirname)
        os.makedirs(odir, exist_ok=True)

        return odir

    def test_train_model(self, tmpdir_factory, particles, ctf, indices):
        """Train the initial heterogeneous model."""

        outdir = self.get_outdir(tmpdir_factory, particles, indices, ctf)
        args = [
            particles.path,
            "-o",
            outdir,
            "--zdim",
            "4",
            "--lr",
            ".0001",
            "--hypervolume-dim",
            "128",
            "--hypervolume-layers",
            "3",
            "--pe-dim",
            "8",
            "--t-extent",
            "4.0",
            "--t-n-grid",
            "2",
            "--num-epochs",
            "3",
            "--epochs-pose-search",
            "1",
            "--n-imgs-pretrain",
            "100",
            "--no-analysis",
            "--log-heavy-interval",
            "1",
        ]
        if ctf.path is not None:
            args += ["--ctf", ctf.path]
        if indices.path is not None:
            args += ["--ind", indices.path]

        parser = argparse.ArgumentParser()
        abinit.add_args(parser)
        abinit.main(parser.parse_args(args))
        assert not os.path.exists(os.path.join(outdir, "analyze.2"))

    @pytest.mark.parametrize(
        "epoch, vol_start_index",
        [(3, 1), (2, 1)],
        ids=["epoch.3,volstart.1", "epoch.2,volstart.1"],
    )
    def test_analyze(
        self, tmpdir_factory, particles, ctf, indices, epoch, vol_start_index
    ):
        """Produce standard analyses for a particular epoch."""

        outdir = self.get_outdir(tmpdir_factory, particles, indices, ctf)
        parser = argparse.ArgumentParser()
        analyze.add_args(parser)
        analyze.main(
            parser.parse_args(
                [
                    outdir,
                    str(epoch),  # Epoch number to analyze - 1-indexed
                    "--pc",
                    "3",  # Number of principal component traversals to generate
                    "--ksample",
                    "10",  # Number of kmeans samples to generate
                    "--vol-start-index",
                    str(vol_start_index),
                ]
            )
        )

        kmeans_dir = os.path.join(outdir, f"analysis_{epoch}", "kmeans10")
        for i in range(vol_start_index, 10 + vol_start_index):
            assert os.path.exists(os.path.join(kmeans_dir, f"vol_{i:03d}.mrc"))
        assert not os.path.exists(
            os.path.join(kmeans_dir, f"vol_{(10 + vol_start_index):03d}.mrc")
        )
