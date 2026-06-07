"""Canonical dataset loaders.

Static registry mapping paper-mentioned dataset names to verified, currently-
working loader recipes. Agents are bound to these recipes by implement_baseline
so they cannot regress to stale tutorial defaults (e.g. load_dataset('imdb')
which fails on modern HuggingFace Hub).

Each recipe carries:
- aliases: case-insensitive names the paper might use
- canonical_import: the Python import line(s)
- canonical_loader: the load expression
- fallback_mirrors: ordered list of fallback URLs / loaders when primary fails
- normalization_stats: per-channel mean/std for image datasets (paper-grade)
- license_note: visible to operator + agent prompt
- notes: extra guidance surfaced in the agent prompt
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class DatasetRecipe:
    canonical_name: str
    aliases: tuple[str, ...]
    canonical_import: str
    canonical_loader: str
    fallback_mirrors: tuple[str, ...] = ()
    normalization_stats: dict | None = None
    license_note: str = ""
    notes: str = ""
    # PR-ξ fields: knowledge-channel metadata (all default-empty for backward compat)
    recipe_id: str = ""                    # stable id, e.g. "dataset.frey_face"
    severity: str = "preferred"            # "strict" | "preferred" | "advisory"
    banned_literals: tuple[str, ...] = ()  # known-bad URL / import patterns
    helper_name: str = ""                  # python identifier for the curated loader
    helper_body: str = ""                  # full def source; rendered into _openresearch_curated.py


DATASET_RECIPES: tuple[DatasetRecipe, ...] = (
    DatasetRecipe(
        canonical_name="IMDB",
        aliases=("imdb", "imdb-reviews", "imdb bow", "imdb sentiment", "imdb-1k"),
        canonical_import="from datasets import load_dataset",
        canonical_loader="load_dataset('stanfordnlp/imdb')",
        notes=(
            "HuggingFace Hub canonical id is owner/name. "
            "Bare 'imdb' is deprecated and raises HfUriError on modern hub."
        ),
    ),
    DatasetRecipe(
        canonical_name="MNIST",
        aliases=("mnist", "mnist-handwritten"),
        canonical_import="from torchvision import datasets, transforms",
        canonical_loader=(
            "datasets.MNIST(root, train=True, download=True, "
            "transform=transforms.ToTensor())"
        ),
        fallback_mirrors=(
            "https://ossci-datasets.s3.amazonaws.com/mnist/",
            "https://yann.lecun.com/exdb/mnist/",
        ),
        normalization_stats={"mean": [0.1307], "std": [0.3081]},
    ),
    DatasetRecipe(
        canonical_name="CIFAR-10",
        aliases=("cifar-10", "cifar10", "cifar 10"),
        canonical_import="from torchvision import datasets, transforms",
        canonical_loader=(
            "datasets.CIFAR10(root, train=True, download=True, "
            "transform=transforms.Compose([transforms.ToTensor(), "
            "transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))"
        ),
        normalization_stats={
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616],
        },
    ),
    DatasetRecipe(
        canonical_name="CIFAR-100",
        aliases=("cifar-100", "cifar100"),
        canonical_import="from torchvision import datasets, transforms",
        canonical_loader=(
            "datasets.CIFAR100(root, train=True, download=True, "
            "transform=transforms.Compose([transforms.ToTensor(), "
            "transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]))"
        ),
        normalization_stats={
            "mean": [0.5071, 0.4865, 0.4409],
            "std": [0.2673, 0.2564, 0.2762],
        },
    ),
    DatasetRecipe(
        canonical_name="ImageNet",
        aliases=("imagenet", "ilsvrc-2010", "ilsvrc-2012", "imagenet-1k"),
        canonical_import="from torchvision import datasets",
        canonical_loader=(
            "datasets.ImageNet(root, split='train', download=False)"
            "  # download=False — ImageNet requires manual setup"
        ),
        license_note=(
            "ImageNet/ILSVRC requires registration. "
            "Download via image-net.org; not auto-downloadable."
        ),
    ),
    DatasetRecipe(
        canonical_name="Frey Face",
        aliases=("frey face", "frey", "freyfaces", "frey_face"),
        canonical_import="import urllib.request, pickle",
        canonical_loader=(
            "pickle.loads(urllib.request.urlopen("
            "'https://raw.githubusercontent.com/y0ast/Variational-Autoencoder/master/freyfaces.pkl'"
            ", timeout=60).read(), encoding='latin1').reshape(-1, 28, 20)"
        ),
        fallback_mirrors=(
            "https://raw.githubusercontent.com/y0ast/Variational-Autoencoder/master/freyfaces.pkl",
            "https://cs.nyu.edu/~roweis/data/frey_rawface.mat",
        ),
        notes=(
            "The original cs.nyu.edu mirror returns HTTP 403 in many networks; the "
            "github raw mirror above is the canonical replacement and serves a "
            "Python-2 pickle of shape (1965, 560) — MUST load with "
            "pickle.loads(data, encoding='latin1') or you'll hit UnicodeDecodeError "
            "on the first non-ASCII byte. For maximum robustness, wrap the load in: "
            "try: <primary>; except Exception: <fallback>. "
            "If every mirror fails, declare 'frey_face' in data_load_failures and "
            "skip the Frey Face experiment — do NOT substitute a synthetic dataset."
        ),
        recipe_id="dataset.frey_face",
        severity="strict",
        banned_literals=(
            "cs.nyu.edu/~roweis/data/frey_rawface.mat",
            "roweis/data/frey_rawface.mat",
        ),
        helper_name="load_frey_face",
        helper_body=(
            "def load_frey_face():\n"
            "    \"\"\"Load the Frey Face dataset from the canonical GitHub mirror.\"\"\"\n"
            "    import pickle\n"
            "    import urllib.request\n"
            "    _PRIMARY = (\n"
            "        \"https://raw.githubusercontent.com/y0ast/\"\n"
            "        \"Variational-Autoencoder/master/freyfaces.pkl\"\n"
            "    )\n"
            "    _FALLBACKS = [\n"
            "        \"https://raw.githubusercontent.com/y0ast/\"\n"
            "        \"Variational-Autoencoder/master/freyfaces.pkl\",\n"
            "    ]\n"
            "    last_exc = None\n"
            "    for url in [_PRIMARY] + _FALLBACKS:\n"
            "        try:\n"
            "            data = urllib.request.urlopen(url, timeout=60).read()\n"
            "            return pickle.loads(data, encoding=\"latin1\").reshape(-1, 28, 20)\n"
            "        except Exception as exc:\n"
            "            last_exc = exc\n"
            "    raise RuntimeError(\n"
            "        f\"load_frey_face: all mirrors failed — last error: {last_exc}\"\n"
            "    )\n"
        ),
    ),
    DatasetRecipe(
        canonical_name="GLUE",
        aliases=("glue",),
        canonical_import="from datasets import load_dataset",
        canonical_loader="load_dataset('nyu-mll/glue', '<subset>')",
        notes=(
            "Subset is required: 'cola', 'sst2', 'mrpc', 'qqp', "
            "'stsb', 'mnli', 'qnli', 'rte', 'wnli'"
        ),
    ),
    DatasetRecipe(
        canonical_name="SQuAD",
        aliases=("squad", "squad-v1", "squad-1.1"),
        canonical_import="from datasets import load_dataset",
        canonical_loader="load_dataset('rajpurkar/squad')",
    ),
    DatasetRecipe(
        canonical_name="SQuAD 2.0",
        aliases=("squad-v2", "squad-2.0", "squad2"),
        canonical_import="from datasets import load_dataset",
        canonical_loader="load_dataset('rajpurkar/squad_v2')",
    ),
    DatasetRecipe(
        canonical_name="Penn Treebank",
        aliases=("penn treebank", "ptb", "penn-treebank"),
        canonical_import="from datasets import load_dataset",
        canonical_loader="load_dataset('ptb_text_only')",
        license_note=(
            "PTB requires LDC license for raw text; "
            "ptb_text_only is a derivative."
        ),
    ),
    DatasetRecipe(
        canonical_name="WikiText-103",
        aliases=("wikitext-103", "wikitext", "wt103"),
        canonical_import="from datasets import load_dataset",
        canonical_loader="load_dataset('Salesforce/wikitext', 'wikitext-103-v1')",
    ),
    DatasetRecipe(
        canonical_name="COCO",
        aliases=("coco", "ms-coco", "mscoco", "coco-2014", "coco-2017"),
        canonical_import="from pycocotools.coco import COCO",
        canonical_loader="COCO(annotation_file)",
        notes=(
            "Download from cocodataset.org. "
            "annotation_file is a JSON path."
        ),
    ),
    DatasetRecipe(
        canonical_name="TIMIT",
        aliases=("timit",),
        canonical_import="",
        canonical_loader="",
        license_note=(
            "LDC license-gated (~$250). Cannot be auto-downloaded. "
            "Skip the experiment or use a free alternative like Common Voice."
        ),
    ),
    DatasetRecipe(
        canonical_name="Reuters RCV1",
        aliases=("rcv1", "reuters rcv1", "reuters-rcv1"),
        canonical_import="",
        canonical_loader="",
        license_note="NIST/LDC license-gated. Cannot be auto-downloaded.",
    ),
    DatasetRecipe(
        canonical_name="Fashion-MNIST",
        aliases=("fashion-mnist", "fashion_mnist", "fmnist"),
        canonical_import="from torchvision import datasets, transforms",
        canonical_loader=(
            "datasets.FashionMNIST(root, train=True, download=True, "
            "transform=transforms.ToTensor())"
        ),
        normalization_stats={"mean": [0.2860], "std": [0.3530]},
    ),
    DatasetRecipe(
        canonical_name="STL-10",
        aliases=("stl-10", "stl10"),
        canonical_import="from torchvision import datasets, transforms",
        canonical_loader=(
            "datasets.STL10(root, split='train', download=True, "
            "transform=transforms.ToTensor())"
        ),
    ),
    DatasetRecipe(
        canonical_name="SVHN",
        aliases=("svhn", "street view house numbers"),
        canonical_import="from torchvision import datasets, transforms",
        canonical_loader=(
            "datasets.SVHN(root, split='train', download=True, "
            "transform=transforms.ToTensor())"
        ),
    ),
    DatasetRecipe(
        canonical_name="CelebA",
        aliases=("celeba", "celeb-a"),
        canonical_import="from torchvision import datasets",
        canonical_loader="datasets.CelebA(root, split='train', download=True)",
        notes="Requires gdown for download in some torchvision versions.",
    ),
    DatasetRecipe(
        canonical_name="LSUN",
        aliases=("lsun",),
        canonical_import="from torchvision import datasets",
        canonical_loader="datasets.LSUN(root, classes=['bedroom_train'])",
        notes=(
            "Categories: bedroom_train, church_outdoor_train, etc. "
            "Download separately from lsun.cs.princeton.edu."
        ),
    ),
)


def find_recipe(name: str) -> DatasetRecipe | None:
    """Case-insensitive lookup; checks canonical_name + aliases."""
    needle = name.strip().lower()
    for r in DATASET_RECIPES:
        if needle == r.canonical_name.lower():
            return r
        if any(needle == a.lower() for a in r.aliases):
            return r
    return None


def _normalize(s: str) -> str:
    """Lowercase, replace underscores with spaces, collapse runs of whitespace."""
    import re as _re
    return _re.sub(r"\s+", " ", s.lower().replace("_", " ")).strip()


def find_recipes_in_text(text: str) -> list[DatasetRecipe]:
    """Scan text for dataset mentions; return matching recipes in declared order.

    Matching is alias-normalized: underscores and extra whitespace are collapsed
    before substring check so "frey_face" and "Frey  Face" both match the Frey
    Face recipe. Dedup by canonical_name is preserved.
    """
    found: list[DatasetRecipe] = []
    seen: set[str] = set()
    needle_text = _normalize(text)
    for r in DATASET_RECIPES:
        names_to_check = [_normalize(r.canonical_name)] + [_normalize(a) for a in r.aliases]
        for n in names_to_check:
            if n in needle_text and r.canonical_name not in seen:
                found.append(r)
                seen.add(r.canonical_name)
                break
    return found


__all__ = ["DatasetRecipe", "DATASET_RECIPES", "find_recipe", "find_recipes_in_text"]
