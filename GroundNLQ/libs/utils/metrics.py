import json
import numpy as np
import lmdb
import io
import torch
import torch.nn.functional as F
import terminaltables
import math
import matplotlib.pyplot as plt
import PIL.Image as Image
import io

from basic_utils import load_jsonl, load_json


class HIT(object):
    def __init__(
            self,
            dataset_name="ego4d",
            gt_file="data/annotations/nlq_val.json",
            vid_feat_dir = "data/features/video_features",
            text_feat_dir = "data/features/text_features",
            task_type="narr",
            feat_type="vis",
            context_action_order=5,
    ):
        self.context_action_order = context_action_order
        self.dataset_name = dataset_name
        self.gt_file = gt_file
        self.vid_feat_dir = vid_feat_dir
        self.text_feat_dir = text_feat_dir
        self.task_type = task_type
        self.feat_type = feat_type
        self.topK = np.array([1, 5, 10])
        print(self.gt_file)
        if self.dataset_name == "ego4d":
            with open(self.gt_file) as file_id:
                self.gt_dict, self.unique_set = self.load_gt_from_json(json.load(file_id))

    def _get_query_feat_by_qid(self, narr_id):
        dump = self.text_feat_txn.get(narr_id.encode())
        with io.BytesIO(dump) as reader:
            try:
                q_dump = np.load(reader, allow_pickle=True)
            except Exception as e:
                print(e)
                print("Cant load narr_id: ", repr(narr_id))
                exit(1)
            token_feats = np.asarray(q_dump['features'])

        return torch.from_numpy(token_feats)[0]

    @torch.no_grad()
    def load_gt_from_json(self, ground_truth):
        gt_dict = {}
        unique_set = set()
        if self.task_type == "narr":
            if self.feat_type == "vis":
                for clip_id, clip_datum in ground_truth.items():
                    for cluster_data in clip_datum["narration_clusters"]:
                        for cluster in cluster_data:
                            if len(cluster) == 1:
                                unique_set.add(cluster[0])
                    vid_feat = torch.load(f"{self.vid_feat_dir}/{clip_id}.pt")
                    vid_feat = vid_feat.permute(1, 0)
                    for pass_idx, pass_datum in enumerate(clip_datum["narrations"]):
                        passes = []
                        for time in pass_datum["exact_times"]:
                            s, e = time
                            s = math.floor(s * 30/16)
                            e = math.ceil(e * 30/16)
                            if s >= vid_feat.size(1):
                                print(f"Warning: win_l>=vid_feat.size(1): in context_action_gt: {s,e}, {vid_feat.size(0)}")
                                passes.append(torch.zeros(512))
                                continue
                            f = vid_feat[:, s:e+1]
                            feat = f.mean(dim = -1)
                            passes.append(feat)
                        gt_dict[(clip_id, pass_idx)] = passes
            elif self.feat_type == "lang":
                self.text_feat_env = lmdb.open(self.text_feat_dir,map_size=8e8, readonly=True, create=False, max_readers=4096 * 8,
                                        readahead=False)
                self.text_feat_txn = self.text_feat_env.begin(buffers=True)
                for clip_id, clip_datum in ground_truth.items():
                    for cluster_data in clip_datum["narration_clusters"]:
                        for cluster in cluster_data:
                            if len(cluster) == 1:
                                unique_set.add(cluster[0])
                    for pass_idx, pass_datum in enumerate(clip_datum["narrations"]):
                        passes = []
                        for ann_id in pass_datum["annotation_uids"]:
                            feat = self._get_query_feat_by_qid(ann_id)
                            passes.append(feat)
                        gt_dict[(clip_id, pass_idx)] = passes
        elif self.task_type == "goalstep":
            if self.feat_type == "vis":
                for clip_id, clip_datum in ground_truth.items():
                    for cluster in clip_datum["goalstep_clusters"]:
                        if len(cluster) == 1:
                            unique_set.add(cluster[0])
                    vid_feat = torch.load(f"{self.vid_feat_dir}/{clip_id}.pt")
                    vid_feat = vid_feat.permute(1, 0)
                    goalstep_queries = clip_datum["goalstep_queries"]
                    passes = []
                    for time in goalstep_queries["exact_times"]:
                        s, e = time
                        s = math.floor(s * 30/16)
                        e = math.ceil(e * 30/16)
                        if s >= vid_feat.size(1):
                            print(f"Warning: win_l>=vid_feat.size(1): in context_action_gt: {s,e}, {vid_feat.size(0)}")
                            passes.append(torch.zeros(512))
                            continue
                        f = vid_feat[:, s:e+1]
                        feat = f.mean(dim = -1)
                        passes.append(feat)
                    gt_dict[clip_id] = passes
            elif self.feat_type == "lang":
                self.text_feat_env = lmdb.open(self.text_feat_dir,map_size=8e8, readonly=True, create=False, max_readers=4096 * 8,
                                        readahead=False)
                self.text_feat_txn = self.text_feat_env.begin(buffers=True)
                for clip_id, clip_datum in ground_truth.items():
                    for cluster in clip_datum["goalstep_clusters"]:
                        if len(cluster) == 1:
                            unique_set.add(cluster[0])
                    goalstep_queries = clip_datum["goalstep_queries"]
                    passes = []
                    for ann_id in goalstep_queries["annotation_uids"]:
                        feat = self._get_query_feat_by_qid(ann_id)
                        passes.append(feat)
                    gt_dict[clip_id] = passes
        else:
            raise NotImplementedError
                
        return gt_dict, unique_set

    def display_results(self, results, title=None):
        display_data = [
            [f"HIT@{ii},AA{jj}" for ii in self.topK for jj in range(2*self.context_action_order+1)]
        ]
        results *= 100

        display_data.append(
            [
                f"{results[jj][ii]:.02f}"
                for ii in range(len(self.topK))
                for jj in range(2*self.context_action_order+1)
            ]
        )
        table = terminaltables.AsciiTable(display_data, title)
        for ii in range(2*self.context_action_order+1 * len(self.topK)):
            table.justify_columns[ii] = "center"
        return table.table

    def evaluate(self, predictions, verbose=True, model=None):
        """Evalutes the performances."""

        result = [[[] for _ in self.topK] for _ in range(2*self.context_action_order+1)]
        unique_results = [[[] for _ in self.topK] for _ in range(2*self.context_action_order+1)]
        for pred_datum in predictions:
            if self.task_type == "goalstep":
                c_uid, s, e, query_id, pass_idx, subclip_idx = pred_datum["annotation_uid"].split("_")
                annotation_uid = f"{c_uid}_{s}_{e}"
            elif self.task_type == "nlq":
                annotation_uid, query_id = pred_datum["annotation_uid"].split("_")
            elif self.task_type == "narr":
                v_uid, c_uid, narr_pass_index, query_id = pred_datum["annotation_uid"].split("_")
                annotation_uid = f"{v_uid}_{c_uid}_{narr_pass_index}"
                narr_pass_index = int(narr_pass_index)
            query_id = int(query_id)
            
            assert pred_datum["clip_uid"] == c_uid
            if self.task_type == "goalstep":
                key = pred_datum["clip_uid"]
            else:
                key = (pred_datum["clip_uid"], narr_pass_index)
            assert key in self.gt_dict, f"{key} not present!"

            gt_feats = self.gt_dict[key]
            
            device = pred_datum[f"predicted_{self.feat_type}_feats"].device
            narr_tensor = torch.stack(gt_feats).to(device)
            pred_tensor = pred_datum[f"predicted_{self.feat_type}_feats"].T

            cos_sim = F.cosine_similarity(pred_tensor.unsqueeze(1), narr_tensor.unsqueeze(0), dim=2)

            k = min(self.topK.max(), narr_tensor.size(0))

            _, top_k_indices = cos_sim.topk(k, dim=1)

            gt_query_ids = [query_id, *[query_id+i for i in range(1, self.context_action_order+1)], *[query_id-i for i in range(1, self.context_action_order+1)]]
            max_query_ids = narr_tensor.size(0)
            gt_mask = [True if 0 <= i < max_query_ids else False for i in gt_query_ids]

            for j in range(2*self.context_action_order+1):
                for i, k in enumerate(self.topK):
                    if gt_mask[j]:
                        if gt_query_ids[j] in top_k_indices[j, :k]:
                            if f"{annotation_uid}_{gt_query_ids[j]}" in self.unique_set:
                                unique_results[j][i].append(1)
                            result[j][i].append(1)
                        else:
                            if f"{annotation_uid}_{gt_query_ids[j]}" in self.unique_set:
                                unique_results[j][i].append(0)
                            result[j][i].append(0)
        
        mean_results = np.array([[np.mean(r) for r in res] for res in result])
        mean_unique_results = np.array([[np.mean(r) for r in res] for res in unique_results])
        assert mean_results.shape == (2*self.context_action_order+1, len(self.topK))

        total_mean = np.mean(mean_results)
        total_mean_unique = np.mean(mean_unique_results)

        score_str = None
        if verbose:
            score_str = self.display_results(np.copy(mean_results))
            score_str = score_str +"\n"+ self.display_results(np.copy(mean_unique_results))
            print(score_str, flush=True)

        return mean_results, total_mean, mean_unique_results, total_mean_unique, score_str

            
class ReferringRecall(object):
    thresholds = np.array([0.3, 0.5, 0.01])
    topK = np.array([1, 3, 5])
    # gt_file: str = "./ego4d_data/ego4d_nlq_v2_ori_data/nlq_val.json"
    # "./ego4d_data/ego4d_nq_ori_data/nlq_val.json"
    def __init__(
            self,
            dataset_name="ego4d",
            gt_file="./ego4d_data/ego4d_nlq_v2_ori_data/nlq_val.json",
            task_type="nlq",
    ):
        self.dataset_name = dataset_name
        self.gt_file = gt_file
        self.task_type = task_type
        if self.task_type == "nlq":
            self.key = "language_queries"
        elif self.task_type == "nlq_feedback":
            self.key = "nlq_feedbacks"
        elif self.task_type == "goalstep":
            self.key = "goalstep_queries"
        elif self.task_type == "narr":
            self.key = "narrations"
        elif self.task_type == "goalstep_nlq":
            self.key = "goalstep_nlq"
        elif self.task_type == "goalstep_nlq_feedback":
            self.key = "goalstep_nlq_feedbacks"
        elif self.task_type == "hd_epic_nlq":
            self.key = "hd_epic_queries"
        elif self.task_type == "hd_epic_nlq_feedback":
            self.key = "hd_epic_nlq_feedbacks"
        print(self.gt_file)
        if self.dataset_name == "ego4d":
            with open(self.gt_file) as file_id:
                self.gt_dict, self.num_gt_queries = self.load_gt_from_json(json.load(file_id))
        else:
            self.gt_dict = {}
            for d in load_jsonl(self.gt_file):
                ts = d["timestamps"][0]
                self.gt_dict[d['query_id']] = {"clip_start_sec": ts[0], "clip_end_sec": ts[1]}
            self.num_gt_queries = len(self.gt_dict)

    def load_gt_from_json(self, ground_truth):
        gt_dict = {}
        num_gt_queries = 0

        for video_datum in ground_truth["videos"]:
            for clip_datum in video_datum["clips"]:
                clip_uid = clip_datum["clip_uid"]
                for ann_datum in clip_datum["annotations"]:
                    if self.key in ann_datum:
                        if self.task_type == "narr":
                            for narr_pass_index, narr_pass_data in enumerate(ann_datum[self.key]):
                                for i, ann in enumerate(narr_pass_data):
                                    gt_dict[f"{ann_datum['annotation_uid']}_{narr_pass_index}_{i}"] = ann
                                    num_gt_queries += 1
                        else:
                            for ann in ann_datum[self.key]:
                                gt_dict[ann["annotation_uid"]] = ann
                                num_gt_queries += 1

        return gt_dict, num_gt_queries

    def compute_IoU(self, pred, gt):
        """Compute the IoU given predicted and ground truth windows."""
        assert isinstance(pred, list) and isinstance(gt, list)
        pred_is_list = isinstance(pred[0], list)
        gt_is_list = isinstance(gt[0], list)
        if not pred_is_list:
            pred = [pred]
        if not gt_is_list:
            gt = [gt]
        pred, gt = np.array(pred), np.array(gt)
        inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
        inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
        inter = np.maximum(0.0, inter_right - inter_left)
        union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
        union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
        union = np.maximum(0.0, union_right - union_left)
        overlap = 1.0 * inter / union
        if not gt_is_list:
            overlap = overlap[:, 0]
        if not pred_is_list:
            overlap = overlap[0]
        return overlap

    def display_results_anet(self, results, title=None):
        display_data = [
            [f"Rank@{ii}\nmIoU@{jj:.1f}" for ii in self.topK for jj in self.thresholds]
        ]
        results *= 100
        display_data.append(
            [
                f"{results[ii][jj]:.02f}"
                for ii in range(len(self.topK))
                for jj in range(len(self.thresholds))
            ]
        )
        table = terminaltables.AsciiTable(display_data, title)
        for ii in range(len(self.thresholds) * len(self.topK)):
            table.justify_columns[ii] = "center"
        return table.table

    def display_results(self, results, title=None):
        display_data = [
            [f"Rank@{ii}\nmIoU@{jj}" for ii in self.topK for jj in self.thresholds]
        ]
        results *= 100

        display_data.append(
            [
                f"{results[jj][ii]:.02f}"
                for ii in range(len(self.topK))
                for jj in range(len(self.thresholds))
            ]
        )
        table = terminaltables.AsciiTable(display_data, title)
        for ii in range(len(self.thresholds) * len(self.topK)):
            table.justify_columns[ii] = "center"
        return table.table

    def evaluate(self, predictions, verbose=True):
        """Evalutes the performances."""

        results = {}
        average_IoU = []
        num_instances = 0

        for pred_datum in predictions:
            if self.task_type == "goalstep":
                raise NotImplementedError
                annotation_uid, s, e, query_id = pred_datum["annotation_uid"].split("_")
                annotation_uid = f"{annotation_uid}_{s}_{e}"
            # elif self.task_type == "nlq":
            #     annotation_uid, query_id = pred_datum["annotation_uid"].split("_")
            # elif self.task_type == "nlq_feedback":
            #     query_id = pred_datum["annotation_uid"].split("__")[-1]
            #     annotation_uid = "__".join(pred_datum["annotation_uid"].split("__")[:-1])
            # elif self.task_type == "narr":
            #     v_uid, c_uid, narr_pass_index, query_id = pred_datum["annotation_uid"].split("_")
            #     annotation_uid = f"{v_uid}_{c_uid}"
            # elif self.task_type == "goalstep_nlq":
            #     annotation_uid, s, e, task_type, query_id = pred_datum["annotation_uid"].split("_")
            #     annotation_uid = f"{annotation_uid}_{s}_{e}"
            # elif self.task_type == "goalstep_nlq_feedback":
            #     annotation_uid, s, e, task_type, query_id = pred_datum["annotation_uid"].split("_")
            #     annotation_uid = f"{annotation_uid}_{s}_{e}"
            # elif self.task_type == "hd_epic_nlq":
            #     annotation_uid, s, e, task_type, query_id = pred_datum["annotation_uid"].split("_")
            #     annotation_uid = f"{annotation_uid}_{s}_{e}"
            # elif self.task_type == "hd_epic_nlq_feedback":
            #     annotation_uid, s, e, task_type, query_id = pred_datum["annotation_uid"].split("_")
            #     annotation_uid = f"{annotation_uid}_{s}_{e}"
            annotation_uid = pred_datum["annotation_uid"]
            key = annotation_uid
            assert key in self.gt_dict, f"{key} not present in gt_dict {self.gt_dict.keys()}"
            gt_query_datum = self.gt_dict[key]
            # if self.task_type == "narr":
            #     gt_query_datum = gt_datum[self.key][int(narr_pass_index)][query_id]
            # else:
            #     gt_query_datum = gt_datum[self.key][query_id]

            # Compute overlap and recalls.
            overlap = self.compute_IoU(
                pred_datum["predicted_times"],
                [[gt_query_datum["clip_start_sec"], gt_query_datum["clip_end_sec"]]],
            )
            average_IoU.append(overlap[0])
            if key not in results:
                results[key] = [[[] for _ in self.topK] for _ in self.thresholds]
            for tt, threshold in enumerate(self.thresholds):
                for rr, KK in enumerate(self.topK):
                    results[key][tt][rr].append((overlap > threshold)[:KK].any())
            num_instances += 1

        
        if self.task_type in ["nlq_feedback", "goalstep_nlq_feedback", "hd_epic_nlq_feedback"]:
            nlq_pair_majority_results = [[[] for _ in self.topK] for _ in self.thresholds]
            nlq_pair_max_results = [[[] for _ in self.topK] for _ in self.thresholds]
            nlq_pair_weighted_results = [[[] for _ in self.topK] for _ in self.thresholds]
            pair_keys_results = {}
            for key, res in results.items():
                pair_key = "__".join(key.split("__")[1:3])
                if pair_key not in pair_keys_results:
                    pair_keys_results[pair_key] = [[[] for _ in self.topK] for _ in self.thresholds]
                for tt in range(len(self.thresholds)):
                    for rr in range(len(self.topK)):
                        pair_keys_results[pair_key][tt][rr].extend(res[tt][rr])
            for key, res in pair_keys_results.items():
                for tt, threshold in enumerate(self.thresholds):
                    for rr, KK in enumerate(self.topK):
                        if len(res[tt][rr]) > 0:
                            nlq_pair_majority_results[tt][rr].append(np.mean(res[tt][rr]) > 0.5)
                            nlq_pair_max_results[tt][rr].append(np.max(res[tt][rr]))
                            nlq_pair_weighted_results[tt][rr].append(np.mean(res[tt][rr]))

            print(f"Number of unique NLQ pairs: {len(pair_keys_results)}")


            nlq_majority_results = [[[] for _ in self.topK] for _ in self.thresholds]
            nlq_max_results = [[[] for _ in self.topK] for _ in self.thresholds]
            nlq_weighted_results = [[[] for _ in self.topK] for _ in self.thresholds]
            grouped_nlq_results = {}
            for key, res in results.items():
                nlq_annotation_uid = key.split("__")[1]
                if nlq_annotation_uid not in grouped_nlq_results:
                    grouped_nlq_results[nlq_annotation_uid] = [[[] for _ in self.topK] for _ in self.thresholds]
                for tt in range(len(self.thresholds)):
                    for rr in range(len(self.topK)):
                        grouped_nlq_results[nlq_annotation_uid][tt][rr].extend(res[tt][rr])
            
            print(f"Number of unique NLQs: {len(grouped_nlq_results)}")

            for key, res in grouped_nlq_results.items():
                for tt, threshold in enumerate(self.thresholds):
                    for rr, KK in enumerate(self.topK):
                        if len(res[tt][rr]) > 0:
                            nlq_majority_results[tt][rr].append(np.mean(res[tt][rr]) > 0.5)
                            nlq_max_results[tt][rr].append(np.max(res[tt][rr]))
                            nlq_weighted_results[tt][rr].append(np.mean(res[tt][rr]))
            
            difficult_nlq_results = [[[] for _ in self.topK] for _ in self.thresholds]
            random_nlq_results = [[[] for _ in self.topK] for _ in self.thresholds]
            for key, res in results.items():
                if "difficult" in key or 'pred-span' in key:
                    for tt in range(len(self.thresholds)):
                        for rr in range(len(self.topK)):
                            difficult_nlq_results[tt][rr].extend(res[tt][rr])
                else:
                    for tt in range(len(self.thresholds)):
                        for rr in range(len(self.topK)):
                            random_nlq_results[tt][rr].extend(res[tt][rr])

            all_results = [[[] for _ in self.topK] for _ in self.thresholds]
            for res in results.values():
                for tt, threshold in enumerate(self.thresholds):
                    for rr, KK in enumerate(self.topK):
                        all_results[tt][rr].extend(res[tt][rr])
            all_results = np.array(all_results)
            mean_results = all_results.mean(axis=-1)
            
            mIoU = np.mean(average_IoU)
            print("NLQ pair majority results shape: ", np.array(nlq_pair_majority_results).shape)
            nlq_pair_majority_results = np.array(nlq_pair_majority_results).mean(axis=-1)
            print("NLQ pair max results shape: ", np.array(nlq_pair_max_results).shape)
            nlq_pair_max_results = np.array(nlq_pair_max_results).mean(axis=-1)
            print("NLQ pair weighted results shape: ", np.array(nlq_pair_weighted_results).shape)
            nlq_pair_weighted_results = np.array(nlq_pair_weighted_results).mean(axis=-1)
            print("NLQ majority results shape: ", np.array(nlq_majority_results).shape)
            nlq_majority_results = np.array(nlq_majority_results).mean(axis=-1)
            print("NLQ max results shape: ", np.array(nlq_max_results).shape)
            nlq_max_results = np.array(nlq_max_results).mean(axis=-1)
            print("NLQ weighted results shape: ", np.array(nlq_weighted_results).shape)
            nlq_weighted_results = np.array(nlq_weighted_results).mean(axis=-1)
            print("Difficult NLQ results shape: ", np.array(difficult_nlq_results).shape)
            difficult_nlq_results = np.array(difficult_nlq_results).mean(axis=-1)
            print("Random NLQ results shape: ", np.array(random_nlq_results).shape)
            random_nlq_results = np.array(random_nlq_results).mean(axis=-1)

            score_str = None
            if verbose:
                print(f"Evaluated: {num_instances} / {self.num_gt_queries} instances")
                score_str = self.display_results(np.copy(nlq_weighted_results))
                print(score_str, flush=True)
                for name, res in [
                    ("mean_results", mean_results),
                    ("nlq_majority_results", nlq_majority_results),
                    ("nlq_max_results", nlq_max_results),
                    ("nlq_pair_weighted_results", nlq_pair_weighted_results),
                    ("nlq_pair_majority_results", nlq_pair_majority_results),
                    ("nlq_pair_max_results", nlq_pair_max_results),
                    ("difficult_nlq_results", difficult_nlq_results),
                    ("random_nlq_results", random_nlq_results),
                ]:
                    print(name)
                    print(self.display_results(np.copy(res)), flush=True)
            
            return mIoU, score_str, mean_results, nlq_weighted_results, nlq_majority_results, nlq_max_results, nlq_pair_weighted_results, nlq_pair_majority_results, nlq_pair_max_results, difficult_nlq_results, random_nlq_results

        all_results = [[[] for _ in self.topK] for _ in self.thresholds]
        for res in results.values():
            for tt, threshold in enumerate(self.thresholds):
                for rr, KK in enumerate(self.topK):
                    all_results[tt][rr].extend(res[tt][rr])
        all_results = np.array(all_results)
        mean_results = all_results.mean(axis=-1)
        mIoU = np.mean(average_IoU)
        score_str = None
        if verbose:
            print(f"Evaluated: {num_instances} / {self.num_gt_queries} instances")
            score_str = self.display_results(np.copy(mean_results))
            print(score_str, flush=True)

        return mean_results, mIoU, score_str
    
    def evaluate_guidelines(self, predictions, verbose=True):
        guidelines = [str(i+1) for i in range(5)]

        per_guideline_results = {
        tmp: [[[] for _ in self.topK] for _ in self.thresholds] for tmp in guidelines
        }
        per_guideline_average_IoU = {tmp: [] for tmp in guidelines}
        num_instances = 0

        for pred_datum in predictions:
            if self.task_type != "nlq_feedback":
                raise ValueError("Task type not supported")
            key = pred_datum["annotation_uid"]
            assert key in self.gt_dict, f"{key} not present!"
            gt_query_datum = self.gt_dict[key]
            guide = gt_query_datum.get("guidelines_followed", "")
            if guide is not None:
                for g in guidelines:
                    if g not in guide:
                        continue

                    # Compute overlap and recalls.
                    overlap = self.compute_IoU(
                        pred_datum["predicted_times"],
                        [[gt_query_datum["clip_start_sec"], gt_query_datum["clip_end_sec"]]],
                    )
                    # per_template_average_IoU[temp].append(np.mean(np.sort(overlap[0])[-3:]))
                    per_guideline_average_IoU[g].append(np.mean(np.sort(overlap[0])[-3:]))

                    for tt, threshold in enumerate(self.thresholds):
                        for rr, KK in enumerate(self.topK):
                            per_guideline_results[g][tt][rr].append(
                            (overlap > threshold)[:KK].any()
                        )
                num_instances += 1

        for tt, threshold in enumerate(self.thresholds):
            for rr, KK in enumerate(self.topK):
                for g in guidelines:
                    print(f"threshold: {threshold}, topK: {KK}, guideline: {g}, len: {len(per_guideline_results[g][tt][rr])}")

        mean_results = {
            k: np.array(v).mean(axis=-1) for k, v in per_guideline_results.items()
        }
        mIoU = {k: np.mean(v) for k, v in per_guideline_average_IoU.items()}

        return mean_results, mIoU
    
    def evaluate_subsets(self, predictions, templates = None, verbose=True):
        """Evalutes the performances."""

        if templates is None:
            templates = [
        'Objects: Where is object X before / after event Y?',
        'Place: Where did I put X?',
        'Objects: Where is object X?',
        'Objects: What did I put in X?',
        'Objects: How many X’s? (quantity question)',
        'Objects: In what location did I see object X ?',
        'Objects: What X did I Y?',
        'Objects: What X is Y?',
        'Objects: State of an object',
        'People: Who did I interact with when I did activity X?',
        ]

        per_template_results = {
        tmp: [[[] for _ in self.topK] for _ in self.thresholds] for tmp in templates
        }
        per_template_average_IoU = {tmp: [] for tmp in templates}
        num_instances = 0

        for pred_datum in predictions:
            # if self.task_type == "goalstep":
            #     annotation_uid, s, e, query_id = pred_datum["annotation_uid"].split("_")
            #     annotation_uid = f"{annotation_uid}_{s}_{e}"
            # elif self.task_type == "nlq":
            #     annotation_uid, query_id = pred_datum["annotation_uid"].split("_")
            # elif self.task_type == "nlq_feedback":
            #     query_id = pred_datum["annotation_uid"].split("__")[-1]
            #     annotation_uid = "__".join(pred_datum["annotation_uid"].split("__")[:-1])
            # query_id = int(query_id)
            key = pred_datum["annotation_uid"]
            assert key in self.gt_dict, f"{key} not present!"
            gt_query_datum = self.gt_dict[key]
            temp = gt_query_datum.get("template", "")
            if temp not in templates:
                continue

            # Compute overlap and recalls.
            overlap = self.compute_IoU(
                pred_datum["predicted_times"],
                [[gt_query_datum["clip_start_sec"], gt_query_datum["clip_end_sec"]]],
            )
            per_template_average_IoU[temp].append(np.mean(np.sort(overlap[0])[-3:]))

            for tt, threshold in enumerate(self.thresholds):
                for rr, KK in enumerate(self.topK):
                    per_template_results[temp][tt][rr].append(
                    (overlap > threshold)[:KK].any()
                )
            num_instances += 1

        mean_results = {
            k: np.array(v).mean(axis=-1) for k, v in per_template_results.items()
        }
        mIoU = {k: np.mean(v) for k, v in per_template_average_IoU.items()}

        return mean_results, mIoU
        

    def _iou(self, candidates, gt):
        start, end = candidates[:, 0].float(), candidates[:, 1].float()
        s, e = gt[0].float(), gt[1].float()
        inter = end.min(e) - start.max(s)
        union = end.max(e) - start.min(s)
        return inter.clamp(min=0) / union

    def evaluate_anet(
            self, submission, verbose=True):

        iou_metrics = torch.tensor(self.thresholds)
        num_iou_metrics = len(iou_metrics)

        recall_metrics = torch.tensor(self.topK)
        max_recall = recall_metrics.max()
        num_recall_metrics = len(recall_metrics)
        recall_x_iou = torch.zeros((num_recall_metrics, len(iou_metrics)))

        for k in submission:
            # print(k)
            gt_grounding = torch.tensor(self.gt_dict[k['query_id']])
            pred_moments = torch.tensor(k["predicted_times"][:max_recall])
            mious = self._iou(pred_moments, gt_grounding)
            mious_len = len(mious)
            bools = mious[:, None].expand(mious_len, num_iou_metrics) > iou_metrics
            for i, r in enumerate(recall_metrics):
                recall_x_iou[i] += bools[:r].any(dim=0)

        recall_x_iou /= len(submission)

        if verbose:
            print(f"Evaluated: {len(submission)} / {self.num_gt_queries} instances")
            score_str = self.display_results_anet(recall_x_iou)
            print(score_str, flush=True)

        return recall_x_iou


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def segment_iou_best(target_segments, candidate_segments):
    """Compute best temporal IoU for each candidate across multiple targets.

    Parameters
    ----------
    target_segments : 2d array (M, 2)
        Temporal target segments containing M x [start, end].
    candidate_segments : 2d array (N, 2)
        Temporal candidate segments containing N x [start, end].

    Returns
    -------
    best_tIoU : 1d array (N,)
        Best IoU score for each candidate across all targets.
    """
    target_segments = np.atleast_2d(target_segments)  # (M, 2)
    candidate_segments = np.atleast_2d(candidate_segments)  # (N, 2)

    # Expand dims for broadcasting
    tt1 = np.maximum(target_segments[:, None, 0], candidate_segments[None, :, 0])
    tt2 = np.minimum(target_segments[:, None, 1], candidate_segments[None, :, 1])

    # Intersection
    inter = (tt2 - tt1).clip(0)

    # Union = len(A) + len(B) - inter
    target_len = (target_segments[:, 1] - target_segments[:, 0])[:, None]  # (M,1)
    candidate_len = (candidate_segments[:, 1] - candidate_segments[:, 0])[None, :]  # (1,N)
    union = target_len + candidate_len - inter

    tIoU = inter.astype(float) / union  # (M, N)

    # Best IoU across targets for each candidate
    best_tIoU = tIoU.max(axis=1)
    return best_tIoU

class SearchDomainEvaluator(object):
    def __init__(self, gt_file, task_type="nlq_feedback"):
        self.task_type = task_type
        if self.task_type == "nlq_feedback":
            self.key = "nlq_feedbacks"
        elif self.task_type == "goalstep_nlq_feedback":
            self.key = "goalstep_nlq_feedbacks"
        elif self.task_type == "hd_epic_nlq_feedback":
            self.key = "hd_epic_nlq_feedbacks"
        else:
            raise NotImplementedError(f"Task type {self.task_type} not supported")

        self.gt_dict, self.num_gt_queries = self.load_gt_from_json(gt_file)

    def load_gt_from_json(self, ground_truth):
        gt_dict = {}
        num_gt_queries = 0
        with open(ground_truth) as file_id:
            ground_truth = json.load(file_id)

        for video_datum in ground_truth["videos"]:
            for clip_datum in video_datum["clips"]:
                clip_uid = clip_datum["clip_uid"]
                for ann_datum in clip_datum["annotations"]:
                    if self.key in ann_datum:
                        for ann in ann_datum[self.key]:
                            gt_dict[ann["annotation_uid"]] = ann
                            num_gt_queries += 1

        return gt_dict, num_gt_queries

    def get_windows_np(self, seq, threshold):
        seq = np.array(seq)
        mask = seq > threshold
        diff = np.diff(mask.astype(int))

        starts = np.where(diff == 1)[0] + 1
        ends   = np.where(diff == -1)[0]

        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends = np.r_[ends, len(seq) - 1]

        # ensure end - start >= 1
        windows = []
        for s, e in zip(starts, ends):
            if e <= s:
                e = s + 1
            windows.append((s, e))

        return windows

    def _gaussian_blur(self, x, k=15, sigma=3):
    # generate 1D Gaussian kernel
        half = k // 2
        t = np.arange(-half, half + 1)
        kernel = np.exp(-(t**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return np.convolve(x, kernel, mode='same')

    def get_gt_removed_ratio(self, results, threshold, smooth=False):
        gt_remain = []
        removed_ratios = []
        for result in results:
            ann_id = result["annotation_uid"]
            pred = np.array(result["pred_saliency"]).flatten()[:result["vlen"]]
            if smooth:
                pred = self._gaussian_blur(pred, k=15, sigma=3)
            gt_query_datum = self.gt_dict[ann_id]
            gt_timestamp_idx = [int(gt_query_datum["clip_start_sec"] * result["fps"]/result["feat_stride"]), int(gt_query_datum["clip_end_sec"] * result["fps"]/result["feat_stride"])]

            thresholded_pred = (pred > threshold).astype(bool)


            if (thresholded_pred[gt_timestamp_idx[0]:gt_timestamp_idx[1]]).all():
                gt_remain.append(1)
            else:
                gt_remain.append(0)

            removed_ratio = 1 - (thresholded_pred.sum() / len(thresholded_pred))
            removed_ratios.append(removed_ratio)

        gt_remain = np.mean(gt_remain)
        removed_ratios = np.mean(removed_ratios)

        return gt_remain, removed_ratios

    def evaluate(self, results, smooth=False):
        """Evaluates the performances."""
        # IOU based metrics
        search_domain_thresholds = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
        iou_metrics = {threshold: [] for threshold in search_domain_thresholds}
        flipped_iou_metrics = {threshold: [] for threshold in search_domain_thresholds}
        merged_iou_metrics = {threshold: [] for threshold in search_domain_thresholds}

        for result in results:
            for threshold in search_domain_thresholds:
                vlen = result["vlen"]
                pred = np.array(result["pred_saliency"]).flatten()[:vlen]
                if smooth:
                    pred = self._gaussian_blur(pred, k=15, sigma=3)
                gt = np.array(result["saliency_labels"]).flatten()[:vlen]
                flipped_pred = 1-pred
                flipped_gt = 1-gt
                pred_windows = self.get_windows_np(pred, threshold=threshold)
                gt_windows = self.get_windows_np(gt, threshold=0.99)
                flipped_pred_windows = self.get_windows_np(flipped_pred, threshold=threshold)
                flipped_gt_windows = self.get_windows_np(flipped_gt, threshold=0.99)
                if len(gt_windows) == 0 or len(flipped_gt_windows) == 0:
                    print(f"Warning: No gt windows found for {result['clip_uid']}, skipping...")
                    continue
                assert len(gt_windows) >0, f"No gt window found: {gt_windows}"
                assert len(flipped_gt_windows) > 0, f"More than one gt window found: {flipped_gt_windows}"
                gt_window = gt_windows[0]
                flipped_gt_window = flipped_gt_windows[0]
                if len(pred_windows) == 0:
                    ious = np.array([0.0])
                else:
                    ious = segment_iou_best(gt_window, np.array(pred_windows))
                    if np.isnan(ious).any():
                        print(f"Warning: NaN ious found for {result['clip_uid']}, pred: {pred}, gt: {gt}, pred_windows: {pred_windows}, gt_window: {gt_window}, ious: {ious}, threshold: {threshold}, vlen: {vlen}, len(pred): {len(result['pred_saliency'])}, len(gt): {len(result['saliency_labels'])}")
                    ious = np.nan_to_num(ious, nan=0.0)
                if len(flipped_pred_windows) == 0:
                    flipped_ious = np.array([0.0])
                else:
                    flipped_ious = segment_iou_best(flipped_gt_window, np.array(flipped_pred_windows))
                    if np.isnan(flipped_ious).any():
                        print(f"Warning: NaN flipped ious found for {result['clip_uid']}, pred: {pred}, gt: {gt}, flipped_pred_windows: {flipped_pred_windows}, flipped_gt_window: {flipped_gt_window}, flipped_ious: {flipped_ious}, threshold: {threshold}, vlen: {vlen}, len(pred): {len(result['pred_saliency'])}, len(gt): {len(result['saliency_labels'])}")
                    flipped_ious = np.nan_to_num(flipped_ious, nan=0.0)
                # assert not NA
                mean_iou = ious.mean()
                iou_metrics[threshold].append(mean_iou)
                mean_flipped_iou = flipped_ious.mean()
                flipped_iou_metrics[threshold].append(mean_flipped_iou)
                merged_iou_metrics[threshold].append(np.mean([mean_iou, mean_flipped_iou]))


        mean_iou_metrics = {k: np.mean(v) for k, v in iou_metrics.items()}
        mean_flipped_iou_metrics = {k: np.mean(v) for k, v in flipped_iou_metrics.items()}
        merged_iou_metrics = {k: np.mean(v) for k, v in merged_iou_metrics.items()}


        gt_remain = {}
        removed_ratios = {}

        for threshold in search_domain_thresholds:
            gt_rem, rem_ratio = self.get_gt_removed_ratio(results, threshold, smooth=smooth)
            gt_remain[threshold] = gt_rem
            removed_ratios[threshold] = rem_ratio


        curve_thresholds = np.arange(0.0, 1.01, 0.01)
        gt_rems = [self.get_gt_removed_ratio(results, t, smooth=smooth) for t in curve_thresholds]
        gt_rem_curve = [item[0] for item in gt_rems]
        removed_ratio_curve = [item[1] for item in gt_rems]

        images = {}
        # curve plots as image array
        plt.switch_backend('Agg')  # prevents GUI backend issues (useful for servers)
        plt.figure(figsize=(6, 6), dpi=200)
        plt.plot(curve_thresholds, gt_rem_curve, linewidth=2)
        plt.xlabel("Search Domain Threshold")
        plt.ylabel("Percent of Ground Truth Remaining")
        plt.title("GT Remaining vs. Search Domain Threshold")
        plt.ylim([0, 1])
        plt.grid(True)
        plt.tight_layout()

        # --- Convert plot to numpy image array ---
        with io.BytesIO() as buf:
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)

            image = Image.open(buf)
            image = np.array(image)
            images['gt_remain_curve'] = image

        plt.figure(figsize=(6, 6), dpi=200)
        plt.plot(curve_thresholds, removed_ratio_curve, linewidth=2)
        plt.xlabel("Search Domain Threshold")
        plt.ylabel("Average Percent of Video Removed")
        plt.title("Average Video Removed vs. Search Domain Threshold")
        plt.ylim([0, 1])
        plt.grid(True)
        plt.tight_layout()

        # --- Convert plot to numpy image array ---
        with io.BytesIO() as buf:
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)

            image = Image.open(buf)
            image = np.array(image)
            images['video_removed_curve'] = image
        
        plt.figure(figsize=(6, 6), dpi=200)
        plt.plot(curve_thresholds, gt_rem_curve, linewidth=2)
        plt.plot(curve_thresholds, removed_ratio_curve, linewidth=2)
        plt.xlabel("Search Domain Threshold")
        plt.ylabel("Percent")
        plt.title("GT Remaining and Video Removed vs. Search Domain Threshold")
        plt.legend(["GT Remaining", "% Video Removed"])
        plt.ylim([0, 1])
        plt.grid(True)
        plt.tight_layout()

        # --- Convert plot to numpy image array ---
        with io.BytesIO() as buf:
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)

            image = Image.open(buf)
            image = np.array(image)
            images['gt_remain_video_removed_curve'] = image
        
        gt_remain_auc = np.trapezoid(gt_rem_curve, curve_thresholds)

        removed_ratio_auc = np.trapezoid(removed_ratio_curve, curve_thresholds)

        return mean_iou_metrics, mean_flipped_iou_metrics, merged_iou_metrics, gt_remain, removed_ratios, gt_remain_auc, removed_ratio_auc, images