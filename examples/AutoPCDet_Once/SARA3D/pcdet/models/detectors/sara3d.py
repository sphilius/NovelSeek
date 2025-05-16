from .detector3d_template import Detector3DTemplate
from ..model_utils.aca_utils import AdaptiveConfidenceAggregation
import torch


class SARA3D(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        
        # Initialize Adaptive Confidence Aggregation module if enabled
        self.use_aca = self.model_cfg.get('USE_ACA', True)
        if self.use_aca:
            self.aca_module = AdaptiveConfidenceAggregation(
                model_cfg=self.model_cfg.get('ACA_CONFIG', {})
            )

    def forward(self, batch_dict):
        # Process through network modules
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        
        # Apply Adaptive Confidence Aggregation if enabled
        if self.use_aca:
            # Check if geometric features are available
            if 'geometric_features' in batch_dict and batch_dict['geometric_features'] is not None:
                try:
                    geometric_features = batch_dict['geometric_features']
                    
                    # Convert to torch tensor if it's numpy array
                    if not isinstance(geometric_features, torch.Tensor):
                        device = next(self.parameters()).device
                        geometric_features = torch.from_numpy(geometric_features).to(device)
                except Exception as e:
                    print(f"Warning: Error processing geometric_features: {e}")
                    # Set to None to use fallback
                    geometric_features = None
                
                for index in range(batch_size):
                    if index in final_pred_dict:
                        pred_boxes = final_pred_dict[index]['pred_boxes']
                        pred_scores = final_pred_dict[index]['pred_scores']
                        
                        # Get geometric features for boxes
                        # This is a simplified approach - in practice, you would need to map
                        # from predicted boxes to the corresponding voxels/points
                        if pred_boxes.shape[0] > 0 and geometric_features.shape[0] > 0:
                            # For simplicity, we'll use a subset of geometric features
                            # In practice, you would need proper mapping from boxes to features
                            num_boxes = pred_boxes.shape[0]
                            num_features = min(num_boxes, geometric_features.shape[0])
                            
                            # Get confidence scores from ACA module
                            box_geometric_features = geometric_features[:num_features]
                            confidence_scores = self.aca_module(box_geometric_features, pred_scores[:num_features])
                            
                            # Apply confidence scores to boxes
                            if num_features < num_boxes:
                                # If we have fewer features than boxes, pad with ones
                                padded_scores = torch.ones_like(pred_scores)
                                padded_scores[:num_features] = confidence_scores
                                confidence_scores = padded_scores
                            
                            # Update scores
                            final_pred_dict[index]['pred_scores'] = confidence_scores
                            final_pred_dict[index]['pred_boxes'][:, 7] = confidence_scores
            else:
                # If geometric features are not available, we can still try to compute them
                # from the predicted boxes and point cloud data
                for index in range(batch_size):
                    if index in final_pred_dict:
                        pred_boxes = final_pred_dict[index]['pred_boxes']
                        pred_scores = final_pred_dict[index]['pred_scores']
                        
                        if pred_boxes.shape[0] > 0:
                            # Create simple geometric features based on box properties
                            # This is a fallback when proper geometric features are not available
                            box_sizes = pred_boxes[:, 3:6]  # width, length, height
                            box_volumes = box_sizes[:, 0] * box_sizes[:, 1] * box_sizes[:, 2]
                            
                            # Normalize volumes
                            normalized_volumes = box_volumes / (box_volumes.max() + 1e-6)
                            
                            # Create simple geometric features: [density, curvature (set to 0), normal (set to [1,0,0])]
                            try:
                                device = pred_boxes.device
                            except:
                                device = next(self.parameters()).device
                                
                            simple_geometric_features = torch.zeros((pred_boxes.shape[0], 5), device=device)
                            simple_geometric_features[:, 0] = normalized_volumes  # Use volume as density
                            simple_geometric_features[:, 2] = 1.0  # Set x-normal to 1
                            
                            # Apply ACA module with these simple features
                            confidence_scores = self.aca_module(simple_geometric_features, pred_scores)
                            
                            # Update scores
                            final_pred_dict[index]['pred_scores'] = confidence_scores
                            final_pred_dict[index]['pred_boxes'][:, 7] = confidence_scores
        
        # Generate recall statistics
        for index in range(batch_size):
            if index in final_pred_dict:
                pred_boxes = final_pred_dict[index]['pred_boxes']
                
                recall_dict = self.generate_recall_record(
                    box_preds=pred_boxes,
                    recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST
                )

        return final_pred_dict, recall_dict