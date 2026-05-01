import os
import json
from pathlib import Path
import gc

from analyze.probe import get_layer_names, extract_layer_features, train_linear_probe
from analyze.metrics import (
    compute_cka,
    compute_numerical_rank,
    compute_variance,
    # compute_isotropy,
    # compute_eRank,
    # compute_gwa,
)
from dataset.common import DatasetBundle
import torch
from tqdm import tqdm
from typing import Dict, Optional

class Analyzer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.results = None

    def analyze(self, dataset: DatasetBundle, num_classes: int, probe: bool = False, output_dir: Path = None) -> Dict:
        """Perform tunnel effect analysis."""
        output_dir = Path(output_dir) if output_dir else None
        if output_dir:
            output_dir.mkdir(exist_ok=True, parents=True)
        os.makedirs(output_dir / f"features", exist_ok=True)

        # Extract features from all layers
        print("Extracting features...")
        layer_names = get_layer_names(self.model)
        # train_feats, train_labels = extract_features(self.model, dataset.train_loader, self.device)
        # test_feats, test_labels = extract_features(self.model, dataset.test_loader, self.device)

        n_layers = len(layer_names)
        print(f"Extracted features from {n_layers} layers.")

        if not probe:
            print("Probe analysis disabled, skipping linear probe training and tunnel effect calculation.")
        
        train_accs, test_accs = [], []
        train_ranks, test_ranks = [], []
        train_vars, test_vars = [], []
        train_ckas, test_ckas = [], []
        isotropy_minmax, isotropy_entropy = [], []
        eranks = []
        for layer in layer_names:
            print(f"Analyzing {layer}...")
            
            print(f"  Extracting test features for {layer}...")
            test_feats, test_labels = extract_layer_features(self.model, layer, dataset.test_loader, self.device)

            print("finish extraction")
            
            if probe:
                print(f"  Extracting train features for {layer}...")
                train_feats, train_labels = extract_layer_features(self.model, layer, dataset.train_loader, self.device)
                
                # Train linear probe on this layer's features
                train_acc, test_acc = train_linear_probe(
                    train_feats,
                    test_feats,
                    train_labels,
                    test_labels,
                    num_classes=num_classes,
                    device=self.device
                )
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                print(f"  {layer}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

                del train_feats, train_labels

            # Compute numerical rank of this layer's features
            print("  Computing numerical rank...")
            # train_ranks.append(compute_numerical_rank(train_feats, device=self.device, n_random_features=8000))
            test_ranks.append(compute_numerical_rank(test_feats, device=self.device, n_random_features=8000))

            # Compute variance explained by top PCA components
            print("  Computing variance explained...")
            # train_vars.append(compute_variance(train_feats, train_labels))
            test_vars.append(compute_variance(test_feats, test_labels))

            # # Compute isotropy and effective rank
            # print("  Computing isotropy and effective rank...")
            # minmax, ent = compute_isotropy(test_feats)
            # isotropy_minmax.append(minmax)
            # isotropy_entropy.append(ent)
            # eranks.append(compute_eRank(test_feats))
            
            print("  Saving features to disk...")
            # torch.save(train_feats, output_dir / f"features/train_feats_{layer}.pt")
            torch.save(test_feats, output_dir / f"features/test_feats_{layer}.pt")

            del test_feats, test_labels
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()


        # Compute gradient-weight alignment (GWA) once for this model/epoch using a limited number of batches
        # try:
        #     print("Computing GWA (gradient-weight alignment)...")
        #     max_batches = min(20, len(dataset.train_loader)) if hasattr(dataset, 'train_loader') else 20
        #     gwa = compute_gwa(self.model, dataset.train_loader, self.device, max_batches=max_batches)
        # except Exception:
        #     print("Failed to compute GWA; recording empty results.")
        #     gwa = {}
        if probe:
            final_acc    = test_accs[-1]
            threshold_95 = 0.95 * final_acc
            threshold_98 = 0.98 * final_acc
            tunnel_95 = next((i for i, a in enumerate(test_accs) if a >= threshold_95), n_layers - 1)
            tunnel_98 = next((i for i, a in enumerate(test_accs) if a >= threshold_98), n_layers - 1)

            print(f"\n  ── Tunnel Analysis ──")
            print(f"  Final acc       : {final_acc:.4f}")
            print(f"  Tunnel start (95%): layer {tunnel_95 + 1} / {n_layers}  "
                f"({100*(tunnel_95+1)/n_layers:.0f}% extractor)")
            print(f"  Tunnel start (98%): layer {tunnel_98 + 1} / {n_layers}  "
                f"({100*(tunnel_98+1)/n_layers:.0f}% extractor)")
            
            # # Compute CKA similarity to input features
            # for i, layer1 in enumerate(layer_names):
            #     cols = []
            #     for j, layer2 in enumerate(layer_names):
            #         feat1 = torch.load(output_dir / f"features/test_feats_{layer1}.pt")
            #         feat2 = torch.load(output_dir / f"features/test_feats_{layer2}.pt")
            #         cka_value = compute_cka(feat1, feat2)
            #         cols.append(cka_value)
            #         del feat1, feat2
            #     test_ckas.append(cols)

        self.results = {
            "layers": layer_names,
            "probe_train_acc": train_accs,
            "probe_test_acc": test_accs,
            # "train_numerical_rank": train_ranks,
            "test_numerical_rank": test_ranks,
            # "train_variance": train_vars,
            "test_variance": test_vars,
            # "isotropy_minmax": isotropy_minmax,
            # "isotropy_entropy": isotropy_entropy,
            # "eRank": eranks,
            # "train_cka": train_ckas,
            # "test_cka": test_ckas,
            "tunnel_start_95": tunnel_95 + 1 if probe else None,
            "tunnel_start_98": tunnel_98 + 1 if probe else None,
            # "gwa": gwa,
        }

        with open(output_dir / "analysis.json", "w") as f:
            json.dump(self.results, f, indent=2)