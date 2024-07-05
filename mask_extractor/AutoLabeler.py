import torch
import numpy as np
import shortuuid

class AutoLabeler():
    def __init__(self, mask_exctractor,train_on_segments, IoU_th=0.01, Conf_th=0.5, topk=5, use_conf_th= True, ow_task = 'task1', use_ucr=True, ucr_th=0.9, pretrained_dataset='scannet200'):
        self.IoU_th = IoU_th
        self.Conf_th = Conf_th
        self.use_conf_th = use_conf_th
        self.topk = topk
        self.train_on_segments = train_on_segments
        self.mask_exctractor = mask_exctractor
        self.pretrained_dataset = pretrained_dataset
        if pretrained_dataset == 'scannet200':
            self.ukn_cls = 200
        elif pretrained_dataset == 'scannet':
            self.ukn_cls = 18
        elif pretrained_dataset == 'stpls3d':
            self.ukn_cls = 15
        self.ow_task = ow_task
        self.use_ucr = use_ucr
        self.ucr_th = ucr_th
    
    def __call__(self, x, target, raw_coord):
        self.indices = []
        self.device = target[0]['labels'].device
        
        self.forward_pass(x, target, raw_coord)
        for b in range(len(target)):
            target = self.Auto_Labeling_bid(b, self.output, target,raw_coord, point2segment=[target[i]['point2segment'] for i in range(len(target))], return_ukn_idxs=False)
        return target

    def Auto_Labeling_bid(self, batch_idx, output, target, raw_coord,point2segment, return_ukn_idxs=True):
        
        IoU_matrix = self.get_IoU(target[batch_idx]['segment_mask'][(target[batch_idx]['labels']!=253)*(target[batch_idx]['labels']!=198)],output['pred_masks'][batch_idx].to(self.device))
        min_id = sum([point2segment[i].shape[0] for i in range(batch_idx)]) if batch_idx!=0 else 0
        max_id = min_id+point2segment[batch_idx].shape[0]
        
        if self.train_on_segments:
            masks = output['pred_masks'][batch_idx][target[batch_idx]['point2segment']].to(self.device)
        else:
            masks = output['pred_masks'][batch_idx]
        pred_logits = output['pred_logits'].to(self.device)
        pred_logits = torch.functional.F.softmax(
            pred_logits ,
            dim=-1)[..., :-1]
        
        scores= self.get_scores(
                pred_logits[batch_idx],
                masks)
        scores = scores.detach()

        max_IoU_per_GT = torch.max(IoU_matrix, dim=0).values
        unk_mask = max_IoU_per_GT<self.IoU_th
        IoU_indices = torch.where(unk_mask)[0]
        #append unknown class pseudo labels
        if self.ow_task == 'task1' or (self.pretrained_dataset != 'scannet200'):
            target[batch_idx]['segment_mask'] = torch.cat((target[batch_idx]['segment_mask'],output['pred_masks'][batch_idx].permute(1,0)[IoU_indices][scores[IoU_indices] > self.Conf_th].to(self.device) > 0), dim=0)
            target[batch_idx]['scores'] = torch.cat((torch.ones_like(target[batch_idx]['labels']),scores[IoU_indices][scores[IoU_indices] > self.Conf_th].to(self.device)))
            target[batch_idx]['labels'] = torch.cat((target[batch_idx]['labels'],(torch.ones_like(IoU_indices)*self.ukn_cls)[scores[IoU_indices] > self.Conf_th].to(self.device).long()))
            
        elif self.use_ucr:
            th = self.ucr_th
            #append known class pseudo labels
            mask__ = torch.zeros((scores.shape[0])).bool()
            mask__[IoU_indices] = True
            known_class_scores = scores[mask__]
            known_pseudo_labels = (torch.max(pred_logits[batch_idx], dim=-1).indices)[mask__]
            target[batch_idx]['segment_mask'] = torch.cat((target[batch_idx]['segment_mask'],output['pred_masks'][batch_idx].permute(1,0)[mask__][known_class_scores > th] > 0), dim=0)
            target[batch_idx]['scores'] = torch.cat((torch.ones_like(target[batch_idx]['labels']),known_class_scores[(known_class_scores > th)]))
            target[batch_idx]['labels'] = torch.cat((target[batch_idx]['labels'],known_pseudo_labels[(known_class_scores > th)].long()))
            unknown_cls_proposals_masks = output['pred_masks'][batch_idx].permute(1,0)[mask__][(known_class_scores < th)*(known_class_scores > 0.5)] > 0
            prev_known_cls_proposals_masks = output['pred_masks'][batch_idx].permute(1,0)[mask__][known_class_scores > th] > 0
            
            if (unknown_cls_proposals_masks.shape[0] != 0) and (prev_known_cls_proposals_masks.shape[0] != 0):
                IoU_matrix_1 = self.get_IoU(prev_known_cls_proposals_masks, unknown_cls_proposals_masks.T)
                max_IoU_per_GT_1 = torch.max(IoU_matrix_1, dim=0).values
                unk_mask_1 = max_IoU_per_GT_1<0.05
                kn_mask_1 = ~unk_mask_1
                if kn_mask_1.shape[0] != 0:
                    target[batch_idx]['segment_mask'] = torch.cat((target[batch_idx]['segment_mask'],unknown_cls_proposals_masks[kn_mask_1]), dim=0)
                    target[batch_idx]['scores'] = torch.cat((target[batch_idx]['scores'],known_class_scores[(known_class_scores < th)*(known_class_scores > 0.5)][kn_mask_1]))
                    target[batch_idx]['labels'] = torch.cat((target[batch_idx]['labels'],(known_pseudo_labels[(known_class_scores < th)*(known_class_scores > 0.5)][kn_mask_1]).long()))
                if unk_mask_1.shape[0] != 0:
                    target[batch_idx]['segment_mask'] = torch.cat((target[batch_idx]['segment_mask'],unknown_cls_proposals_masks[unk_mask_1]), dim=0)
                    target[batch_idx]['scores'] = torch.cat((target[batch_idx]['scores'],known_class_scores[(known_class_scores < th)*(known_class_scores > 0.5)][unk_mask_1]))
                    target[batch_idx]['labels'] = torch.cat((target[batch_idx]['labels'],(torch.ones_like(known_class_scores[(known_class_scores < th)*(known_class_scores > 0.5)][unk_mask_1])*self.ukn_cls).long()))
            elif unknown_cls_proposals_masks.shape[0] != 0:
                target[batch_idx]['segment_mask'] = torch.cat((target[batch_idx]['segment_mask'],unknown_cls_proposals_masks), dim=0)
                target[batch_idx]['scores'] = torch.cat((target[batch_idx]['scores'],known_class_scores[(known_class_scores < th)*(known_class_scores > 0.5)]))
                target[batch_idx]['labels'] = torch.cat((target[batch_idx]['labels'],(torch.ones_like(known_class_scores[(known_class_scores < th)*(known_class_scores > 0.5)])*self.ukn_cls).long()))
                
        else:
            th = 0.5
            mask__ = torch.zeros((scores.shape[0])).bool()
            mask__[IoU_indices] = True
            known_class_scores = scores[mask__]
            known_pseudo_labels = (torch.max(pred_logits[batch_idx], dim=-1).indices)[mask__]
            target[batch_idx]['segment_mask'] = torch.cat((target[batch_idx]['segment_mask'],output['pred_masks'][batch_idx].permute(1,0)[mask__][known_class_scores > th] > 0), dim=0)
            target[batch_idx]['scores'] = torch.cat((torch.ones_like(target[batch_idx]['labels']),known_class_scores[(known_class_scores > th)]))
            target[batch_idx]['labels'] = torch.cat((target[batch_idx]['labels'],known_pseudo_labels[(known_class_scores > th)].long()))

        return target
    
    def get_IoU(self, GT_segment_mask, P_segment_mask):
        threshold = 0.5
        if GT_segment_mask.nelement() !=0 :
            intersection = GT_segment_mask.float()@(torch.nn.Sigmoid()(P_segment_mask)>threshold).float()
            union = torch.stack(tuple(torch.sum((GT_segment_mask[inst_id,:].float()+(torch.nn.Sigmoid()(P_segment_mask)>threshold).float().T),dim=1)-intersection[inst_id,:] for inst_id in range(GT_segment_mask.shape[0])), dim = 0)
            IoU = intersection/union
        else:
            IoU = torch.zeros(P_segment_mask.shape[-1])
        return IoU
    
    def get_scores(self, mask_cls, mask_pred):
        scores_per_query = torch.max(mask_cls, dim = 1).values
        result_pred_mask = (mask_pred > 0).float()
        heatmap = mask_pred.float().sigmoid()
        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)
        score = scores_per_query * mask_scores_per_image
        return score
    
    def forward_pass(self, x, target, raw_coord):
        with torch.no_grad():
            self.output = self.mask_exctractor.to(self.device)(x,
                                    point2segment=[target[i]['point2segment'] for i in range(len(target))],
                                    raw_coordinates=raw_coord, is_eval=True)
    def return_indices(self, bid):
        return self.indices[bid]
                