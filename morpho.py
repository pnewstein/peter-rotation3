"Tests out if I can quanitativly compare the morphology of neurons"

from pathlib import Path

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import navis
PATH = Path("/storage/archive/sylwestrak/Peter-Zeiss/Peter-Zeiss/20220510_m1583-6-1-000.swc")
J_PATH = Path("/storage/archive/sylwestrak/j/")
VIRUS_PATH = Path("/storage/archive/sylwestrak/raw_confocal/20220525/20220525_m1583-2.czi-000.swc")
PVT_PATH = Path("/home/petern/Documents/rtalk3/f/f-000.swc")


def preprocess_swc(path: Path):
    """ensures that the swc at a path is compatible with navis"""
    data = path.read_text("utf-8")
    # remove tab delimiters
    if "\t" in data:
        out = data.replace("\t", " ")
        path.write_text(out, "utf-8")

def get_skeletons(directory: Path) -> list[navis.core.skeleton.TreeNeuron]:
    "finds all the swc files in a directory and returns the sceletons"
    # prepare all swcs 
    swcs = [p for p in directory.iterdir() if p.suffix == ".swc"]
    for path in swcs:
        preprocess_swc(path)
    skeletons = [navis.read_swc(p) for p in swcs]
    return skeletons

def rotate_neuron(x: float, y: float, z: float, neuron: navis.core.skeleton.TreeNeuron) -> navis.core.skeleton.TreeNeuron:
    "Applies a rotation to a neurons and returns a copy"
    out_neuron = neuron.copy(deepcopy=True)
    nodes_matrix = out_neuron.nodes.loc[:, ["x", "y", "z"]]
    rotated = R.from_rotvec([x, y, z], degrees=True).apply(nodes_matrix)
    out_neuron.nodes.loc[:, ["x", "y", "z"]] = rotated
    out_neuron._clear_temp_attr()
    return out_neuron

def make_clusters(skeletons: list[navis.core.skeleton.TreeNeuron]):
    "Heiraricaly cluster a few neurons"
    dps_list = [navis.make_dotprops(s) for s in skeletons]
    aba = navis.nblast_allbyall(dps_list, progress=False)
    aba_name_key = {str(uuid): skel.name for uuid, skel in zip(aba.columns, skeletons)}
    # make matrix symmetrical
    aba_mean = (aba + aba.T) / 2
    # convert similariteis to distances
    aba_dist = (1 - aba_mean)
    # convert to vector-form
    aba_vec = squareform(aba_dist)
    Z = linkage(aba_vec, method="ward")
    fig, ax = plt.subplots(tight_layout=True)
    dn = dendrogram(Z, labels=aba_mean.columns, ax=ax, link_color_func=lambda _: "blue")
    ax.set_xticklabels([aba_name_key[uuid.get_text()] for uuid in ax.get_xticklabels()], rotation=30, ha='right')
    print(aba)


def make_svgs(directory: Path):
    "Makes svgs from all the skeletons in the directory"
    skeletons = get_skeletons(directory)
    for skeleton in skeletons:
        fig, ax = plt.subplots(tight_layout=True)#, subplot_kw={"projection": "3d"})
        navis.plot2d(skeleton, "2d", ax=ax, color="k")
        ax.set_aspect("equal")
        # save as a svg with the same name as .swc
        plt.savefig((directory / skeleton.name).with_suffix(".svg"))
        # save a bit of ram
        fig.clear()

def test_rotation():
    neurons = get_skeletons(PATH.parent)
    rotated = rotate_neuron(0, 0, 90, neurons[0])
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    rotated.plot2d(method="2d", ax=axs[0])
    neurons[0].plot2d(method="2d", ax=axs[1])
    axs[0].set_aspect("equal")
    make_clusters(neurons)
    rotated = neurons[0]
    make_clusters(neurons)
    plt.show()

def main():
    make_svgs(PVT_PATH.parent)


if __name__ == '__main__':
    main()
