
class Baseline(nn.Module):
    '''
    Hierarchical Concepts As Bottleneck

    x -> s -> c -> y

    '''
    def __init__(self,  
                        in_channels, # input channel of the image
                        out_channels, # number of categories to recognize
                        vis_clue_num, # number of visual clues 
                        concept_num, # number of concepts
                        seg_backbone=None, # backbone of x -> s
                        include_background = True, # whether include the background as visual clues 
                        concept_backbone='resnet18', # backbone of s -> c
                        concept_act = torch.sigmoid, # activation function of concepts
                        expand_dim=1024, # latent dimension for c -> y
                        desc_dim = 128, # dimension of concept descritor
                        group_num = 8, # number of groups
                        head = 256 # number of combinations, group_num * head = extra concept number
                        ):
        super().__init__()

        if seg_backbone is None:
            self.x_to_s = nn.Identity()
            self.use_sem = False
            vis_clue_num = 1
        else:
            self.x_to_s = seg_backbone
            self.use_sem = True
            self.vis_clue_num = vis_clue_num
        
        if not include_background:
            vis_clue_channel_num = in_channels * (vis_clue_num - 1)
        else:
            vis_clue_channel_num = in_channels * vis_clue_num

        self.include_background = include_background
        
        if concept_backbone == 'resnet18':
            self.s_to_c = resnet(vis_clue_channel_num, concept_num, depth=18, weights=None)
        elif concept_backbone == 'resnet50':
            self.s_to_c = resnet(vis_clue_channel_num, concept_num, depth=50, weights=None)
        else:
            raise NotImplementedError()

        self.concept_act = concept_act

        if desc_dim:
            # describe concepts with vectors
            assert desc_dim >= 4,f'{desc_dim} is not valid'

            self.encoder = nn.Sequential(
                        Conv2D1x1(1, desc_dim//4),
                        Conv2D1x1(desc_dim//4, desc_dim//2),
                        Conv2D1x1(desc_dim//2, desc_dim)
            )
            self.decoder = nn.Sequential(
                        Conv1D1x1(desc_dim, desc_dim//2),
                        Conv1D1x1(desc_dim//2, desc_dim//4),
                        Conv1D1x1(desc_dim//4, 1, activation=None)                       
            )
            self.use_dsc = True

        else:
            self.use_dsc = False
        
        extra_concepts = group_num * head # number of extra concepts

        if extra_concepts:
            # activate concept interaction
            self.grouping = nn.Sequential(
                Conv1x1(concept_num, concept_num*2),
                Conv1x1(concept_num*2, concept_num*2),
                Conv1x1(concept_num*2, extra_concepts)
            )
            self.add_concept = True
            self.group_num = group_num
            self.head = head
        else:
            self.add_concept = False

        if expand_dim:
            self.fc = nn.Sequential(
            Conv1D1x1(concept_num+extra_concepts, expand_dim),
            Conv1D1x1(expand_dim, out_channels, activation=None)        
            )
        else:
            self.fc = Conv1D1x1(concept_num+extra_concepts, out_channels, activation=None)

    def forward(self, x):
        image = x['image']

        # x -> s
        if self.use_sem:
            sem_logits = self.x_to_s({'image':image})['logit'] # (B, C, H, W)
            if sem_logits.size(1) == 1:
                assign = torch.sigmoid(sem_logits)
            else:
                assign = torch.softmax(sem_logits, dim=1)
                if (not self.include_background) and (assign.size(1) > 1):
                    assign = assign[:,1:,...]
            vis_clues = (assign.unsqueeze(2)*image.unsqueeze(1)).flatten(1, 2)

        else:
            vis_clues = image
        # s -> c
        concept_logit = self.s_to_c(vis_clues)
        concept_pred = self.concept_act(concept_logit)

        concept = concept_pred

        # c -> y
        if self.add_concept:
            raw_concept = concept.clone()
            square_concept = concept**2

            groups = self.grouping(concept)
            groups = nn.functional.gumbel_softmax(groups.reshape(concept.size(0), self.group_num, self.head, -1), dim=1, hard=True)

            # grouping concepts
            concept = concept.unsqueeze(1).unsqueeze(1)
            concept = (groups * concept).sum(-1).flatten(-2)

            # group squared concepts
            square_concept = (groups * square_concept.unsqueeze(1).unsqueeze(1)).sum(-1).flatten(-2)

            concept = (concept**2 - square_concept) / 2 # (x1+x2)**2 - x1**2 - x2**2
            concept = torch.cat((raw_concept, concept), dim=-1)

        if self.use_dsc:
            # use vector to represent concepts
            concept_desc = self.encoder(concept.unsqueeze(1).unsqueeze(1))
            concept_desc = concept_desc.squeeze(2)
        else:
            # use scalar to represent concepts
            concept_desc = concept.unsqueeze(1) 
        
        concept_desc = concept_desc.transpose(1,2).contiguous()

        logit = self.fc(concept_desc)
        
        if self.use_dsc:
            logit = logit.transpose(1,2).contiguous()
            logit = self.decoder(logit).squeeze(1)
        else:
            logit = logit.squeeze(-1)
        
        out = {
            'logit': logit,
            'concept_logit': concept_logit
        }

        return out                

class LinearProbe(nn.Module):
    '''
    Hierarchical Concepts As Bottleneck

    x -> s -> c -> y

    '''
    def __init__(self,  
                        out_channels, # number of categories to recognize
                        concept_num, # number of concepts
                        expand_dim=1024, # latent dimension for c -> y
                        group_num = 1, # number of groups
                        concept_intervention=False,
                        ):
        super().__init__()
        extra_concepts = group_num

        if extra_concepts:
            # activate concept interaction
            self.grouping = nn.Sequential(
                Conv1x1(concept_num, concept_num*2),
                Conv1x1(concept_num*2, concept_num*2),
                Conv1x1(concept_num*2, extra_concepts*concept_num)
            )
            self.add_concept = True
            self.group_num = group_num
        else:
            self.add_concept = False

        if expand_dim:
            self.fc = nn.Sequential(
            Conv1D1x1(concept_num+extra_concepts, expand_dim),
            Conv1D1x1(expand_dim, out_channels, activation=None)        
            )
        else:
            self.fc = Conv1D1x1(concept_num+extra_concepts, out_channels, activation=None)

        self.concept_interv = concept_intervention

    def forward(self, x):
        
        if self.training:
            concept = x['concept'].cuda()
        else:
            if self.concept_interv:
                concept = x['concept'].cuda()
            else:
                concept = x['pred_concept'].cuda()

        # c -> y
        if self.add_concept:
            raw_concept = concept.clone()
            square_concept = concept**2

            groups = self.grouping(concept)
            groups = nn.functional.relu(groups.reshape(concept.size(0), self.group_num, -1))
 
            concept = concept.unsqueeze(-1)
            groups = groups.expand(concept.size(0), self.group_num, raw_concept.shape[-1])
            concept = (groups@concept).flatten(-2) # B, H, G, 1

            # group squared concepts
            square_concept = square_concept.unsqueeze(-1)
            square_concept = (groups**2@square_concept).flatten(-2)
            concept = (concept**2 - square_concept) / 2 # (x1+x2)**2 - x1**2 - x2**2
            # (a+b)
            norminal = (groups.sum(-1))**2
            # (a2+b2)
            square_norm = (groups**2).sum(-1)
            norm = (norminal - square_norm)/2

            norm = torch.clamp(norm, min=1e-9)
            concept = concept / norm
            concept = torch.relu(concept)
            concept = torch.sqrt(concept)
            concept = torch.cat((raw_concept, concept), dim=-1)

        concept_desc = concept.unsqueeze(1) 
        
        concept_desc = concept_desc.transpose(1,2).contiguous()

        logit = self.fc(concept_desc)
        
        logit = logit.squeeze(-1)
        
        out = {
            'logit': logit,
            'concept_logit': x['pred_concept']
        }

        return out            

class SemanticBottleneck(nn.Module):
    '''
    Semantic Bottleneck

    x -> s -> y

    '''
    def __init__(self,  
                        in_channels, # input channel of the image
                        out_channels, # number of categories to recognize
                        vis_clue_num, # number of visual clues 
                        seg_backbone=None, # backbone of x -> s
                        include_background = True, # whether include the background as visual clues 
                        concept_backbone='resnet18', # backbone of s -> c
                        vis_intervention=False
                        ):
        super().__init__()

        if seg_backbone is None:
            self.x_to_s = nn.Identity()
            self.use_sem = False
            vis_clue_num = 1
        else:
            self.x_to_s = seg_backbone
            self.use_sem = True
            self.vis_clue_num = vis_clue_num
        
        if not include_background:
            vis_clue_channel_num = in_channels * (vis_clue_num - 1)
        else:
            vis_clue_channel_num = in_channels * vis_clue_num

        self.include_background = include_background
        
        if concept_backbone == 'resnet18':
            self.s_to_y = resnet(vis_clue_channel_num, out_channels, depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        elif concept_backbone == 'resnet50':
            self.s_to_y = resnet(vis_clue_channel_num, out_channels, depth=50, weights="ResNet50_Weights.IMAGENET1K_V1")
        elif concept_backbone == 'inception':
            self.s_to_y = Inception(vis_clue_channel_num, out_channels)
        else:
            raise NotImplementedError()
        
        self.vis_interv = vis_intervention

    def forward(self, x):
        image = x['image']
        # x -> s
        if self.use_sem:
            if self.training:
                mask = x['mask']
                if self.vis_clue_num == 2 and not self.include_background:
                    assign = mask.unsqueeze(1)
                else:
                    if len(mask.shape) == 3:
                        assign = torch.nn.functional.one_hot(mask, num_classes=self.vis_clue_num).permute(0,3,1,2)
                    else:
                        assign = mask.float()
            else:
                mask = x['mask'].to(image.device)
                if self.vis_clue_num == 2 and not self.include_background:
                    assign = mask.unsqueeze(1)
                else:
                    if self.vis_interv:
                        if len(mask.shape) == 3:
                            assign = torch.nn.functional.one_hot(mask, num_classes=self.vis_clue_num).permute(0,3,1,2)
                        else:
                            assign = mask.float()
                    else:
                        with torch.no_grad():
                            self.x_to_s.eval()
                            sem_logits = self.x_to_s({'image':image})['logit'] # (B, C, H, W)
                            assign = torch.softmax(sem_logits, dim=1)
            if (not self.include_background) and (assign.size(1) > 1):
                assign = assign[:,1:,...]
            vis_clues = (assign.unsqueeze(2)*image.unsqueeze(1)).flatten(1, 2)
        else:
            vis_clues = image
        # s -> c
        logit = self.s_to_y(vis_clues)
        out = {
            'logit': logit,
        }

        return out            

class HCBM(nn.Module):
    '''
    Hierarchical Concepts As Bottleneck

    x -> s -> c -> y

    '''
    def __init__(self,  
                        in_channels, # input channel of the image
                        out_channels, # number of categories to recognize
                        vis_clue_num, # number of visual clues 
                        concept_num, # number of concepts
                        seg_backbone=None, # backbone of x -> s
                        include_background = True, # whether include the background as visual clues 
                        concept_backbone='resnet18', # backbone of s -> c
                        concept_act = torch.sigmoid, # activation function of concepts
                        expand_dim=1024, # latent dimension for c -> y
                        desc_dim = 128, # dimension of concept descritor
                        group_num = 8, # number of groups
                        head = 256, # number of combinations, group_num * head = extra concept number
                        vis_intervention=False,
                        concept_intervention=False,
                        ):
        super().__init__()

        if seg_backbone is None:
            self.x_to_s = nn.Identity()
            self.use_sem = False
            vis_clue_num = 1
        else:
            self.x_to_s = seg_backbone
            self.use_sem = True
            self.vis_clue_num = vis_clue_num
        
        if not include_background:
            vis_clue_channel_num = in_channels * (vis_clue_num - 1)
        else:
            vis_clue_channel_num = in_channels * vis_clue_num

        self.include_background = include_background
        
        if concept_backbone == 'resnet18':
            self.s_to_c = ResNet(in_channels=vis_clue_channel_num, out_channels=concept_num, depth=18, weights="ResNet18_Weights.IMAGENET1K_V1")
        elif concept_backbone == 'resnet50':
            self.s_to_c = ResNet(in_channels=vis_clue_channel_num, out_channels=concept_num, depth=50, weights="ResNet50_Weights.IMAGENET1K_V1")
        elif concept_backbone == 'inception':
            self.s_to_c = Inception(vis_clue_channel_num, concept_num)
        else:
            raise NotImplementedError()

        self.s_to_c.load_state_dict(torch.load('/home/manli/3rd_trim_ultrasounds/Concept_Bottleneck_Network/logs/exp_6/model.t7'))
        
        self.concept_act = concept_act
        
        # extra_concepts = group_num * head # number of extra concepts
        extra_concepts = group_num

        if extra_concepts:
            # activate concept interaction
            bin_concept_num = 8
            self.grouping = nn.Sequential(
                Conv1x1(bin_concept_num, bin_concept_num*2),
                Conv1x1(bin_concept_num*2, bin_concept_num*2),
                Conv1x1(bin_concept_num*2, extra_concepts*bin_concept_num)
            )
            self.add_concept = True
            self.group_num = group_num
            self.head = head
        else:
            self.add_concept = False

        if expand_dim:
            self.fc1 = Conv1D1x1((concept_num+extra_concepts), expand_dim, bn=False)
            self.fc2 = Conv1D1x1(expand_dim, out_channels, activation=None)
            # self.fc = nn.Sequential(
            #     Conv1D1x1((concept_num+extra_concepts), expand_dim, bn=False),
            #     Conv1D1x1(expand_dim, out_channels, activation=None)
            # )
            # self.fc = Conv1D1x1(concept_num+extra_concepts, out_channels, activation=None)
        else:
            self.fc = Conv1D1x1(concept_num+extra_concepts, out_channels, activation=None)

        self.vis_interv = vis_intervention
        self.concept_interv = concept_intervention

    def forward(self, x, mask_cervix=False):
        image = x['image']
        emb = None

        # x -> s
        if self.use_sem:
            # if self.training:
            if False:
                mask = x['mask']
                if self.vis_clue_num == 2 and not self.include_background:
                    assign = mask.unsqueeze(1)
                else:
                    if len(mask.shape) == 3:
                        if self.vis_clue_num > 1:
                            assign = torch.nn.functional.one_hot(mask, num_classes=self.vis_clue_num).permute(0,3,1,2)
                        else:
                            assign = mask.unsqueeze(1)
                    else:
                        assign = mask.float()
            else:
                
                if self.vis_clue_num == 2 and not self.include_background:
                    mask = x['mask'].to(image.device)
                    assign = mask.unsqueeze(1)
                else:
                    if self.vis_interv:
                        mask = x['mask'].to(image.device)
                        if len(mask.shape) == 3:
                            if self.vis_clue_num > 1:
                                assign = torch.nn.functional.one_hot(mask, num_classes=self.vis_clue_num).permute(0,3,1,2)
                            else:
                                assign = mask.unsqueeze(1)
                        else:
                            assign = mask.float()
                    else:
                        with torch.no_grad():
                            self.x_to_s.eval()
                            sem_logits = self.x_to_s({'image':image})['coarse_logit'] # (B, C, H, W)
                            if self.vis_clue_num > 1:
                                assign = torch.softmax(sem_logits, dim=1)
                            else:
                                assign = torch.sigmoid(sem_logits)
            if (not self.include_background) and (assign.size(1) > 1):
                assign = assign[:,1:,...]
            vis_clues = (assign.unsqueeze(2)*image.unsqueeze(1)).flatten(1, 2)
        else:
            vis_clues = image

        # s -> c
        with torch.no_grad():
            self.s_to_c.eval()
            concept_logit = self.s_to_c({'image':vis_clues})['logit']
        
        aux_logit = None
        if not isinstance(concept_logit, torch.Tensor):
            aux_logit = concept_logit.aux_logits
            concept_logit = concept_logit.logits

        # mask out concept preds based on segmentation
        mask = torch.ones_like(concept_logit)
        seg_mask = torch.argmax(assign, dim=1)
        for b in range(seg_mask.size(0)):
            # cervix
            if ((1 not in seg_mask[b,...]) + (2 not in seg_mask[b,...]) + (3 not in seg_mask[b,...])) >= 3:
                mask[b, [22, 24, 25, 26]] = 0
            if 4 not in seg_mask[b,...]:
                mask[b, 23] = 0
            
            # femur
            if 5 not in seg_mask[b,...]:
                mask[b, [0,1,2,3]] = 0

            # abdomen
            if 6 not in seg_mask[b,...]:
                mask[b, 5] = 0
            if 8 not in seg_mask[b,...]:
                mask[b, 6] = 0
            if 9 not in seg_mask[b,...]:
                mask[b, 7] = 0
            if 7 not in seg_mask[b, ...]:
                mask[b, [9, 10, 11, 12]] = 0
                mask[b, 8] = 0
            if ((6 not in seg_mask[b,...]) + (7 not in seg_mask[b,...]) + (8 not in seg_mask[b,...])) >= 3:
                mask[b, [4, 8]] = 0
                mask[b, [9, 10, 11, 12]] = 0

            # head
            if 10 not in seg_mask[b,...]:
                mask[b, 14] = 0
            if 11 not in seg_mask[b,...]:
                mask[b, 16] = 0
            if 12 not in seg_mask[b,...]:
                mask[b, 15] = 0
            if 13 not in seg_mask[b, ...]:
                mask[b, [18, 19, 20, 21]] = 0
                mask[b, 17] = 0
            if ((10 not in seg_mask[b,...]) + (12 not in seg_mask[b,...]) + (13 not in seg_mask[b,...])) >= 3:
                mask[b, [13, 17]] = 0
                mask[b, [18, 19, 20, 21]] = 0
        
        concept_mask = mask

        concept_pred = self.concept_act(concept_logit)

        if self.training:
            concept = x['concept']
        else:
            if self.concept_interv:
                concept = x['concept']
            else:
                concept = concept_pred.detach()

        concept_pred = concept_pred.detach() * concept_mask
        concept_mask[:, 23] = 0  # ignore bladder
        if mask_cervix:
            concept_mask[:, 22:27] = 0 # ignore cervix concepts
        # concept[:,[2, 3, 4, 8, 13, 17, 22, 24]] = (concept[:,[2, 3, 4, 8, 13, 17, 22, 24]] > 0.5).float()
        # concept[:,[0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]] = (concept[:,[0, 1, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 25, 26]]*10).long()/10.
        concept = concept * concept_mask
        
        # c -> y
        if self.add_concept:
            raw_concept = concept.clone()
            concept = concept[:,[2, 3, 4, 8, 13, 17, 22, 24]]
            square_concept = concept**2

            groups = self.grouping(concept)
            # groups = nn.functional.gumbel_softmax(groups.reshape(concept.size(0), self.group_num, self.head, -1), dim=1, hard=True)
            groups = nn.functional.relu(groups.reshape(concept.size(0), self.group_num, -1))
            # groups = torch.sigmoid(groups.reshape(concept.size(0), self.group_num, -1))
            # B, G, C
            # B, C

            # grouping concepts
            # groups = groups.permute(0,2,1,3) # B, H, G, C
            # concept = concept.unsqueeze(-1).unsqueeze(-1) # B, C, 1, 1
            # concept = concept.permute(0,2,1,3) # B, 1, C, 1
            concept = concept.unsqueeze(-1)
            groups = groups.expand(concept.size(0), self.group_num, square_concept.shape[-1])
            # concept = concept.repeat(1, self.head, 1, 1)
            # print(groups.shape, concept.shape)
            # concept = (groups * concept).sum(-1).flatten(-2)
            concept = (groups@concept).flatten(-2) # B, H, G, 1

            # group squared concepts
            # square_concept = (groups * square_concept.unsqueeze(1).unsqueeze(1)).sum(-1).flatten(-2)
            square_concept = square_concept.unsqueeze(-1)
            # square_concept = square_concept.permute(0,2,1,3)
            square_concept = (groups**2@square_concept).flatten(-2)
            concept = (concept**2 - square_concept) / 2 # (x1+x2)**2 - x1**2 - x2**2
            # (a+b)
            norminal = (groups.sum(-1))**2
            # (a2+b2)
            square_norm = (groups**2).sum(-1)
            norm = (norminal - square_norm)/2

            norm = torch.clamp(norm, min=1e-9)
            concept = concept / norm
            concept = torch.relu(concept)
            concept = torch.sqrt(concept)
            concept = torch.cat((raw_concept, concept), dim=-1)

        concept = concept.unsqueeze(-1)
        emb = self.fc1(concept)
        logit = self.fc2(emb)
        # logit = self.fc(concept)
        logit = logit.squeeze(-1)
        
        out = {
            'logit': logit,
            'concept_logit': concept_logit,
            'full_concept_logit': concept,
            'concept_mask': concept_mask,
            'concept_pred': concept_pred,
            'emb': emb.squeeze(-1).detach()
        }
        if aux_logit is not None:
            out['aux_logit'] = aux_logit

        return out            
