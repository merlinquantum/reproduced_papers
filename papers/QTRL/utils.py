import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hypothèse : Les modules MerLin sont importés ici dans ton environnement réel
# from merlin import CircuitBuilder, QuantumLayer, LexGrouping

def required_qubits_estimation(model):
    numpy_weights = {}
    nw_list = [] 
    nw_list_normal = []
    for name, param in model.state_dict().items():
        numpy_weights[name] = param.cpu().numpy()
    for i in numpy_weights:
        nw_list.append(list(numpy_weights[i].flatten()))
    for i in nw_list:
        for j in i:
            nw_list_normal.append(j)
    print("# of NN parameters: ", len(nw_list_normal))
    
    n_qubits = int(np.ceil(np.log2(len(nw_list_normal))))
    print("Required qubit number: ", n_qubits)
    
    return n_qubits, nw_list_normal

### Some tool function definition ###########
def probs_to_weights(probs_, model):
    new_state_dict = {}
    data_iterator = probs_.view(-1)

    for name, param in model.state_dict().items():
        shape = param.shape
        num_elements = param.numel()
        chunk = data_iterator[:num_elements].reshape(shape)
        new_state_dict[name] = chunk
        data_iterator = data_iterator[num_elements:]
        
    return new_state_dict

def generate_qubit_states_torch(n_qubit):
    # Create a tensor of shape (2**n_qubit, n_qubit) with all possible combinations of 0 and 1
    all_states = torch.cartesian_prod(*[torch.tensor([-1.0, 1.0]) for _ in range(n_qubit)])
    return all_states

def apply_layer(x, layer_config, state_dict, device, dtype):
    layer_type = layer_config["type"]
    name       = layer_config["name"]
    
    ## Layers ## 
    if layer_type == "Conv2d":
        weight = state_dict[f'{name}.weight'].to(device).type(dtype)
        bias = state_dict[f'{name}.bias'].to(device).type(dtype) if f'{name}.bias' in state_dict else None
        x = F.conv2d(x, weight, bias, **layer_config["params"])
        
    elif layer_type == "MaxPool2d":
        x = F.max_pool2d(x, **layer_config["params"])
        
    elif layer_type == "Linear":
        weight = state_dict[f'{name}.weight'].to(device).type(dtype)
        bias = state_dict[f'{name}.bias'].to(device).type(dtype) if f'{name}.bias' in state_dict else None
        x = F.linear(x, weight, bias)
    
    ## Flatten operation ##
    elif layer_type == "Flatten":
        x = x.view(x.size(0), -1)
    
    ## Activation function ##
    elif layer_type == "ReLU":
        x = F.relu(x)
    elif layer_type == "Sigmoid":
        x = F.sigmoid(x)
    elif layer_type == "Tanh":
        x = F.tanh(x)
    elif layer_type == "LeakyReLU":
        x = F.leaky_relu(x, negative_slope=0.01)
    elif layer_type == "Softmax":
        x = F.softmax(x, dim=-1)
    elif layer_type == "Softplus":
        x = F.softplus(x)
        
    return x

def network_config_extract(model):
    network_config = []
    for name, layer in model.named_modules():
        layer_type = str(type(layer)).split('.')[-1].split("'")[0]  # Extracting the layer type
        config = {"type": layer_type, "name": name, "params": {}}
        
        # Handling kernel_size
        if hasattr(layer, 'kernel_size'):
            kernel_size = layer.kernel_size
            if type(kernel_size) == int:
                config["params"]["kernel_size"] = kernel_size
                
        # Handling stride
        if hasattr(layer, 'stride'):
            stride = layer.stride
            if type(stride) == int:
                config["params"]["stride"] = stride
            elif type(stride) != int:
                config["params"]["stride"] = stride[0]
                
        network_config.append(config)
    return network_config

# ==========================================
# HYBRID MODEL INTEGRATION
# ==========================================
def QuantumTrain(
        model,
        n_qubit,
        nw_list_normal,
        n_blocks,
        device,
        network_config,
        gamma = 0.1,
        beta  = 0.8,
        alpha = 0.6,
        mm_arch = [4, 20, 4],
):
    
    class LewHybridNN(nn.Module):
        
        class QLayer(nn.Module):
                    def __init__(self, n_blocks):
                        super().__init__()
                        self.n_blocks = n_blocks
                        self.n_qubits = int(np.ceil(np.log2(len(nw_list_normal))))
                        self.n_modes = 2 * self.n_qubits
                        self.n_photons = self.n_qubits
        
                        self.Circuit = CircuitBuilder(self.n_modes)
                        self.Circuit.add_entangling_layer(trainable=True, name='U_3_1')
                        
                        self.Circuit.add_angle_encoding(modes=[i for i in range(self.n_modes)], name="input")
                        
                        self.Circuit.add_entangling_layer(trainable=True, name='U_3_2')
                        for _ in range(self.n_blocks): 
                            self.Circuit.add_entangling_layer(trainable=True)
        
                        self.layer1 = QuantumLayer(
                            input_size=self.n_modes,
                            builder=self.Circuit,
                            n_photons=self.n_photons,
                            dtype=torch.float32,
                        )
                        self.layer2 = LexGrouping(self.layer1.output_size, len(nw_list_normal))
                            
                    def forward(self, q_input): 
                        # L'entrée arrive maintenant de l'extérieur de la classe !
                        return self.layer2(self.layer1(q_input))
        class MappingModel(nn.Module):
            def __init__(self, input_size, hidden_sizes, output_size): 
                super().__init__()
                self.input_layer = nn.Linear(input_size, hidden_sizes[0])
                self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
                self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
                
            def forward(self, X):
                X = X.type_as(self.input_layer.weight)
                X = F.relu(self.input_layer(X))
                for hidden in self.hidden_layers:
                    X = F.relu(hidden(X))
                output = self.output_layer(X)
                return output

        def __init__(self):
            super().__init__()
            self.MappingNetwork = self.MappingModel(n_qubit + 1, mm_arch, 1).to(device)  
            self.QuantumNN = self.QLayer(n_blocks).to(device) 
            self.easy_scale_coeff = 2**(n_qubit - 1)
        
        def forward(self, x):
            device = x.device

            # ==========================================
            # DÉFINITION DE L'ENTRÉE QUANTIQUE (À l'extérieur de QLayer)
            # ==========================================
            batch_size = x.shape[0]
            
            # -> Si tu veux l'Option A (Pas de dépendance à l'image) :
            q_input = torch.zeros(batch_size, self.QuantumNN.n_modes, device=device)
            
            # -> Si tu veux l'Option B (Hyper-réseau), il faudrait faire ça :
            # (À condition que x soit de taille n_modes, sinon il faut un padding)
            # q_input = x 
            # ==========================================

            # On passe explicitement l'entrée au circuit quantique
            probs_ = self.QuantumNN(q_input)
            
            probs_ = probs_.view(batch_size, -1)[:, :len(nw_list_normal)]
            
            qubit_states_torch = generate_qubit_states_torch(n_qubit)[:len(nw_list_normal)].to(device)
            qubit_states_batch = qubit_states_torch.unsqueeze(0).expand(batch_size, -1, -1)
            
            probs_col = probs_.unsqueeze(-1)
            
            combined_data_torch = torch.cat((qubit_states_batch, probs_col), dim=-1)
            combined_data_torch = combined_data_torch.reshape(batch_size * len(nw_list_normal), n_qubit + 1)
            
            prob_val_post_processed = self.MappingNetwork(combined_data_torch)
            prob_val_post_processed = (beta * torch.tanh(gamma * self.easy_scale_coeff * prob_val_post_processed))**(alpha) 
            prob_val_post_processed = prob_val_post_processed.view(batch_size, len(nw_list_normal))
            prob_val_post_processed = prob_val_post_processed - prob_val_post_processed.mean(dim=1, keepdim=True)
            
            # NOTE : probs_to_weights ne gère pas nativement les batchs multiples (batch_size > 1). 
            # L'implémentation suppose batch_size = 1.
            state_dict = probs_to_weights(prob_val_post_processed[0], model)
            
            dtype = torch.float32
            for layer in network_config:
                x = apply_layer(x, layer, state_dict, device, dtype)
            
            return x
        
    return LewHybridNN




#Test

# ==========================================
# SCRIPT DE TESTS D'INTÉGRATION
# (À coller à la fin de ton fichier actuel)
# ==========================================

if __name__ == "__main__":
    print("--- DÉBUT DU TEST D'INTÉGRATION ---")

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")

    # 1. Création d'un petit modèle classique factice (pour tester vite)
    # Total des poids : (4*8)+8 + (8*2)+2 = 58 paramètres
    dummy_model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    # 2. Extraction de la configuration
    n_qubit, nw_list_normal = required_qubits_estimation(dummy_model)
    network_config = network_config_extract(dummy_model)

    # 3. Création de la CLASSE hybride
    LewHybridNN_class = QuantumTrain(
        model=dummy_model,
        n_qubit=n_qubit,
        nw_list_normal=nw_list_normal,
        n_blocks=2, # On teste avec 2 blocs d'intrication
        device=device,
        network_config=network_config
    )

    # 4. INSTANCIATION du modèle hybride
    hybrid_model = LewHybridNN_class().to(device)

    # 5. Création d'une donnée d'entrée factice (ex: un batch de 1 élément de taille 4)
    x_input = torch.randn(1, 4).to(device)

    # ==========================================
    # TEST 1 : LE FORWARD PASS
    # ==========================================
    print("\n[Test 1] Exécution du Forward Pass (Calcul des prédictions)...")
    try:
        output = hybrid_model(x_input)
        print(f"✅ Succès ! L'image a traversé le réseau. Dimension de sortie : {output.shape}")
    except Exception as e:
        print(f"❌ ÉCHEC lors du Forward :\n{e}")

    # ==========================================
    # TEST 2 : LE BACKWARD PASS (Vérification des gradients)
    # ==========================================
    print("\n[Test 2] Exécution du Backward Pass (Vérification de l'Autograd)...")
    try:
        # On simule le calcul d'une "loss" (erreur) basique
        loss = output.sum()
        # On déclenche la rétropropagation (ça calcule les dérivées de tout le modèle)
        loss.backward()

        # On va fouiller dans la couche quantique MerLin pour voir si les gradients sont arrivés
        has_gradients = False
        for name, param in hybrid_model.QuantumNN.layer1.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                print(f"✅ Gradient détecté dans MerLin (Paramètre: {name})")
                break # Dès qu'on en trouve un, c'est que la connexion marche

        if has_gradients:
            print("✅ Succès absolu ! L'erreur remonte bien de la sortie classique jusqu'au circuit quantique.")
        else:
            print("❌ ÉCHEC : Le circuit MerLin est déconnecté du graphe PyTorch (Aucun gradient reçu).")

    except Exception as e:
        print(f"❌ ÉCHEC lors du Backward :\n{e}")

    print("\n--- FIN DES TESTS ---")# ==========================================
# SCRIPT DE TESTS D'INTÉGRATION
# (À coller à la fin de ton fichier actuel)
# ==========================================

if __name__ == "__main__":
    print("--- DÉBUT DU TEST D'INTÉGRATION ---")

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")

    # 1. Création d'un petit modèle classique factice (pour tester vite)
    # Total des poids : (4*8)+8 + (8*2)+2 = 58 paramètres
    dummy_model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    # 2. Extraction de la configuration
    n_qubit, nw_list_normal = required_qubits_estimation(dummy_model)
    network_config = network_config_extract(dummy_model)

    # 3. Création de la CLASSE hybride
    LewHybridNN_class = QuantumTrain(
        model=dummy_model,
        n_qubit=n_qubit,
        nw_list_normal=nw_list_normal,
        n_blocks=2, # On teste avec 2 blocs d'intrication
        device=device,
        network_config=network_config
    )

    # 4. INSTANCIATION du modèle hybride
    hybrid_model = LewHybridNN_class().to(device)

    # 5. Création d'une donnée d'entrée factice (ex: un batch de 1 élément de taille 4)
    x_input = torch.randn(1, 4).to(device)

    # ==========================================
    # TEST 1 : LE FORWARD PASS
    # ==========================================
    print("\n[Test 1] Exécution du Forward Pass (Calcul des prédictions)...")
    try:
        output = hybrid_model(x_input)
        print(f"✅ Succès ! L'image a traversé le réseau. Dimension de sortie : {output.shape}")
    except Exception as e:
        print(f"❌ ÉCHEC lors du Forward :\n{e}")

    # ==========================================
    # TEST 2 : LE BACKWARD PASS (Vérification des gradients)
    # ==========================================
    print("\n[Test 2] Exécution du Backward Pass (Vérification de l'Autograd)...")
    try:
        # On simule le calcul d'une "loss" (erreur) basique
        loss = output.sum()
        # On déclenche la rétropropagation (ça calcule les dérivées de tout le modèle)
        loss.backward()

        # On va fouiller dans la couche quantique MerLin pour voir si les gradients sont arrivés
        has_gradients = False
        for name, param in hybrid_model.QuantumNN.layer1.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                print(f"✅ Gradient détecté dans MerLin (Paramètre: {name})")
                break # Dès qu'on en trouve un, c'est que la connexion marche

        if has_gradients:
            print("✅ Succès absolu ! L'erreur remonte bien de la sortie classique jusqu'au circuit quantique.")
        else:
            print("❌ ÉCHEC : Le circuit MerLin est déconnecté du graphe PyTorch (Aucun gradient reçu).")

    except Exception as e:
        print(f"❌ ÉCHEC lors du Backward :\n{e}")

    print("\n--- FIN DES TESTS ---")
