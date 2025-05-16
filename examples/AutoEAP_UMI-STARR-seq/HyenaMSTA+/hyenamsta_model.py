import tensorflow as tf
import keras
import keras.layers as kl
from keras_nlp.layers import SinePositionEncoding, TransformerEncoder

class EnhancedHyenaPlusLayer(kl.Layer):
    """
    Enhanced Hyena+DNA layer with multi-scale feature extraction, residual connections,
    explicit dimension alignment, and layer normalization for improved gradient flow and stability.
    """
    def __init__(self, filters, kernel_size, output_dim, use_residual=True, dilation_rate=1, 
                 kernel_regularizer=None, **kwargs):
        super(EnhancedHyenaPlusLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.use_residual = use_residual
        self.dilation_rate = dilation_rate
        self.kernel_regularizer = kernel_regularizer
        
        # Core convolution for long-range dependencies with mild regularization
        self.conv = kl.Conv1D(filters, kernel_size, padding='same', 
                             kernel_regularizer=kernel_regularizer)
        
        # Multi-scale feature extraction with dilated convolutions
        self.dilated_conv = kl.Conv1D(filters // 2, kernel_size, 
                                     padding='same', 
                                     dilation_rate=dilation_rate,
                                     kernel_regularizer=kernel_regularizer)
        
        # Parallel small kernel convolution for local features
        self.local_conv = kl.Conv1D(filters // 2, 3, padding='same',
                                   kernel_regularizer=kernel_regularizer)
        
        # Batch normalization and activation
        self.batch_norm = kl.BatchNormalization()
        self.activation = kl.Activation('relu')
        
        # Feature fusion layer
        self.fusion = kl.Dense(filters, kernel_regularizer=kernel_regularizer)
        
        # Explicit dimension alignment projection with regularization
        self.projection = kl.Dense(output_dim, kernel_regularizer=kernel_regularizer)
        
        # Layer normalization for stability
        self.layer_norm = kl.LayerNormalization()
        
        # Input projection for residual connection if dimensions don't match
        self.input_projection = None
        if use_residual:
            self.input_projection = kl.Dense(output_dim, kernel_regularizer=kernel_regularizer)
    
    def call(self, inputs, training=None):
        # Save input for residual connection
        residual = inputs
        
        # Process through main convolution
        x_main = self.conv(inputs)
        
        # Process through dilated convolution for capturing long-range patterns
        x_dilated = self.dilated_conv(inputs)
        
        # Process through local convolution for capturing local patterns
        x_local = self.local_conv(inputs)
        
        # Concatenate multi-scale features
        x_multi = tf.concat([x_dilated, x_local], axis=-1)
        
        # Fuse features
        x = self.fusion(x_multi) + x_main
        
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        
        # Project to target dimension
        x = self.projection(x)
        
        # Add residual connection if enabled
        if self.use_residual:
            # Project input if needed for dimension matching
            residual = self.input_projection(residual)
            x = x + residual
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        return x
    
    def get_config(self):
        config = super(EnhancedHyenaPlusLayer, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'output_dim': self.output_dim,
            'use_residual': self.use_residual,
            'dilation_rate': self.dilation_rate,
            'kernel_regularizer': self.kernel_regularizer
        })
        return config

class HybridContextAwareMSTA(kl.Layer):
    """
    Hybrid Context-Aware Motif-Specific Transformer Attention (HCA-MSTA) module
    with enhanced biological interpretability and selective motif attention.
    Combines the strengths of previous approaches with improved positional encoding.
    """
    def __init__(self, num_motifs, motif_dim, num_heads=4, dropout_rate=0.1, 
                 kernel_regularizer=None, activity_regularizer=None, **kwargs):
        super(HybridContextAwareMSTA, self).__init__(**kwargs)
        self.num_motifs = num_motifs
        self.motif_dim = motif_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = kernel_regularizer
        self.activity_regularizer = activity_regularizer
        
        # Motif embeddings with mild regularization
        self.motif_embeddings = self.add_weight(
            shape=(num_motifs, motif_dim),
            initializer='glorot_uniform',
            regularizer=activity_regularizer,
            trainable=True,
            name='motif_embeddings'
        )
        
        # Positional encoding for motifs
        self.motif_position_encoding = self.add_weight(
            shape=(num_motifs, motif_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='motif_position_encoding'
        )
        
        # Biological prior weights for motifs (importance weights)
        self.motif_importance = self.add_weight(
            shape=(num_motifs, 1),
            initializer='ones',
            regularizer=activity_regularizer,
            trainable=True,
            name='motif_importance'
        )
        
        # Attention mechanism components with regularization
        self.query_dense = kl.Dense(motif_dim, kernel_regularizer=kernel_regularizer)
        self.key_dense = kl.Dense(motif_dim, kernel_regularizer=kernel_regularizer)
        self.value_dense = kl.Dense(motif_dim, kernel_regularizer=kernel_regularizer)
        
        # Multi-head attention
        self.attention = kl.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=motif_dim // num_heads,
            dropout=dropout_rate
        )
        
        # Gating mechanism
        self.gate_dense = kl.Dense(motif_dim, activation='sigmoid', 
                                  kernel_regularizer=kernel_regularizer)
        
        # Output projection
        self.output_dense = kl.Dense(motif_dim, kernel_regularizer=kernel_regularizer)
        self.dropout = kl.Dropout(dropout_rate)
        self.layer_norm = kl.LayerNormalization()
        
        # Feed-forward network for feature enhancement
        self.ffn_dense1 = kl.Dense(motif_dim * 2, activation='relu', 
                                  kernel_regularizer=kernel_regularizer)
        self.ffn_dense2 = kl.Dense(motif_dim, kernel_regularizer=kernel_regularizer)
        self.ffn_layer_norm = kl.LayerNormalization()
        self.ffn_dropout = kl.Dropout(dropout_rate)
    
    def positional_masking(self, sequence_embeddings, motif_embeddings):
        """
        Generate hybrid positional masking based on sequence and motif relevance
        with improved biological context awareness and motif importance weighting.
        Combines inverse distance and Gaussian approaches for better biological relevance.
        """
        # Calculate similarity between sequence embeddings and motif embeddings
        similarity = tf.matmul(sequence_embeddings, tf.transpose(motif_embeddings, [0, 2, 1]))
        
        # Scale similarity scores for numerical stability
        scaled_similarity = similarity / tf.sqrt(tf.cast(self.motif_dim, tf.float32))
        
        # Apply softmax to get attention-like weights
        attention_weights = tf.nn.softmax(scaled_similarity, axis=-1)
        
        # Calculate position-aware weights with hybrid approach
        seq_length = tf.shape(sequence_embeddings)[1]
        motif_length = tf.shape(motif_embeddings)[1]
        
        # Create position indices
        position_indices = tf.range(seq_length)[:, tf.newaxis] - tf.range(motif_length)[tf.newaxis, :]
        position_indices_float = tf.cast(position_indices, tf.float32)
        
        # Inverse distance weighting (for local context)
        inverse_weights = 1.0 / (1.0 + tf.abs(position_indices_float))
        
        # Gaussian weighting (for smooth transitions)
        gaussian_weights = tf.exp(-0.5 * tf.square(position_indices_float / 8.0))  # Gaussian with σ=8
        
        # Combine both weighting schemes for a hybrid approach
        # This captures both sharp local context and smooth transitions
        position_weights = 0.5 * inverse_weights + 0.5 * gaussian_weights
        position_weights = tf.expand_dims(position_weights, 0)  # Add batch dimension
        
        # Apply motif importance weighting with temperature scaling for sharper focus
        motif_weights = tf.nn.softmax(self.motif_importance * 1.5, axis=0)  # Temperature scaling
        motif_weights = tf.expand_dims(tf.expand_dims(motif_weights, 0), 1)  # [1, 1, num_motifs, 1]
        
        # Combine attention weights with position weights and motif importance
        combined_weights = attention_weights * position_weights * tf.squeeze(motif_weights, -1)
        
        return combined_weights
    
    def call(self, inputs, training=None):
        # Add positional encoding to motif embeddings
        batch_size = tf.shape(inputs)[0]
        
        # Expand motif embeddings and position encodings to batch dimension
        motifs = tf.tile(tf.expand_dims(self.motif_embeddings, 0), [batch_size, 1, 1])
        pos_encoding = tf.tile(tf.expand_dims(self.motif_position_encoding, 0), [batch_size, 1, 1])
        
        # Add positional encoding to motifs
        motifs_with_pos = motifs + pos_encoding
        
        # Prepare query from input sequence embeddings
        query = self.query_dense(inputs)
        
        # Prepare key and value from motifs with positional encoding
        key = self.key_dense(motifs_with_pos)
        value = self.value_dense(motifs_with_pos)
        
        # Generate positional masking
        pos_mask = self.positional_masking(query, motifs_with_pos)
        
        # Apply attention with positional masking
        attention_output = self.attention(
            query=query,
            key=key,
            value=value,
            attention_mask=pos_mask,
            training=training
        )
        
        # Apply gating mechanism to selectively focus on relevant features
        gate = self.gate_dense(inputs)
        gated_attention = gate * attention_output
        
        # Process through output projection with residual connection
        output = self.output_dense(gated_attention)
        output = self.dropout(output, training=training)
        output = self.layer_norm(output + inputs)  # Residual connection
        
        # Apply feed-forward network with residual connection
        ffn_output = self.ffn_dense1(output)
        ffn_output = self.ffn_dense2(ffn_output)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        final_output = self.ffn_layer_norm(output + ffn_output)  # Residual connection
        
        return final_output
    
    def get_config(self):
        config = super(HybridContextAwareMSTA, self).get_config()
        config.update({
            'num_motifs': self.num_motifs,
            'motif_dim': self.motif_dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'kernel_regularizer': self.kernel_regularizer,
            'activity_regularizer': self.activity_regularizer
        })
        return config

def HyenaMSTAPlus(params):
    """
    Enhanced HyenaMSTA+ model for enhancer activity prediction with multi-scale feature
    extraction, hybrid attention mechanism, and improved biological context modeling.
    """
    if params['encode'] == 'one-hot':
        input_layer = kl.Input(shape=(249, 4))
    elif params['encode'] == 'k-mer':
        input_layer = kl.Input(shape=(1, 64))
    
    # Regularization settings - milder than previous run
    l2_reg = params.get('l2_reg', 1e-6)
    kernel_regularizer = tf.keras.regularizers.l2(l2_reg)
    activity_regularizer = tf.keras.regularizers.l1(l2_reg/20)
    
    # Hyena+DNA processing
    x = input_layer
    hyena_layers = []
    
    # Number of motifs and embedding dimension - optimized based on previous runs
    num_motifs = params.get('num_motifs', 48)  # Adjusted to optimal value from Run 2
    motif_dim = params.get('motif_dim', 96)    # Adjusted to optimal value from Run 2
    
    # Apply Enhanced Hyena+DNA layers with increasing dilation rates
    for i in range(params['convolution_layers']['n_layers']):
        # Use increasing dilation rates for broader receptive field
        dilation_rate = 2**min(i, 2)  # 1, 2, 4 (capped at 4 to avoid excessive sparsity)
        
        hyena_layer = EnhancedHyenaPlusLayer(
            filters=params['convolution_layers']['filters'][i],
            kernel_size=params['convolution_layers']['kernel_sizes'][i],
            output_dim=motif_dim,
            dilation_rate=dilation_rate,
            kernel_regularizer=kernel_regularizer,
            name=f'EnhancedHyenaPlus_{i+1}'
        )
        x = hyena_layer(x)
        hyena_layers.append(x)
        
        if params['encode'] == 'one-hot':
            x = kl.MaxPooling1D(2)(x)
        
        if params['dropout_conv'] == 'yes':
            x = kl.Dropout(params['dropout_prob'])(x)
    
    # Hybrid Context-Aware MSTA processing
    ca_msta = HybridContextAwareMSTA(
        num_motifs=num_motifs,
        motif_dim=motif_dim,
        num_heads=params.get('ca_msta_heads', 8),
        dropout_rate=params['dropout_prob'],
        kernel_regularizer=kernel_regularizer,
        activity_regularizer=activity_regularizer
    )
    
    x = ca_msta(x)
    
    # Flatten and dense layers
    x = kl.Flatten()(x)
    
    # Fully connected layers
    for i in range(params['n_dense_layer']):
        x = kl.Dense(params['dense_neurons'+str(i+1)],
                     name=str('Dense_'+str(i+1)))(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(params['dropout_prob'])(x)
    
    # Main model bottleneck
    bottleneck = x
    
    # Heads per task (developmental and housekeeping enhancer activities)
    tasks = ['Dev', 'Hk']
    outputs = []
    for task in tasks:
        outputs.append(kl.Dense(1, activation='linear', name=str('Dense_' + task))(bottleneck))
    
    # Build Keras model
    model = keras.models.Model([input_layer], outputs)
    model.compile(
        keras.optimizers.Adam(learning_rate=params['lr']),
        loss=['mse', 'mse'],
        loss_weights=[1, 1]
    )
    
    return model, params
