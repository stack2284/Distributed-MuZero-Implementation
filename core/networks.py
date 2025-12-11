import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class AbstractNetwork(nn.Module):
    def __init__(self) :
        super().__init__()
        self.float() #keeps model in float32 for macos please change for diffrent device

    def representation(self , observation) :
        raise NotImplementedError

    def dynamics (self , encode_state , action ) :
        raise NotImplementedError

    def prediction(self , encoded_state) :
        raise NotImplementedError
    
class MuZeroNetwork (AbstractNetwork) :
    def __init__ (self , observation_shape , action_space_size , action_space_type='discrete') :
        super().__init__()
        self.action_space_type = action_space_type
        # For simple games like CartPole, we just want to predict 1 raw value.
        # self.full_support_size = 601 
        # standard Muzero 601 for catg predction
        self.full_support_size = 1
        if len(observation_shape)  == 3 : # C H W using Resnet
               self.encoded_state_shape = (256, 6, 6)
               #self.representation_net = ResNetRepresentation(observation_shape)
               pass
        else: #vector use MLP
               self.encoded_state_shape = (64,)
               self.representation_net = MLPRepresentation(observation_shape[0] , self.encoded_state_shape[0])


        #dynamic network (predict next state + reward )

        self.dynamics_net = DynamicsNetwork(
            self.encoded_state_shape[0],
            action_space_size,
            self.encoded_state_shape[0]
        )

        self.prediction_net  = PredictionNetwork(
                self.encoded_state_shape[0],
                action_space_size ,
                self.full_support_size
        )
        self.to(DEVICE)
    def representation (self , observation) :
            #    #ensuring input on mps for mac training
            #    observation = torch.tensor(observation , dtype=torch.float32 , device=DEVICE)
            #    observation = observation.unsqueeze(0)
            #    return self.representation_net(observation)
            if isinstance(observation, torch.Tensor):
                observation = observation.clone().detach().to(dtype=torch.float32, device=DEVICE)
            else:
                observation = torch.tensor(observation, dtype=torch.float32, device=DEVICE)

            if observation.dim() == 1:
                observation = observation.unsqueeze(0)

            return self.representation_net(observation)

    def dynamics (self , encoded_state , action) :
            if isinstance(action, torch.Tensor):
                action = action.clone().detach().to(dtype=torch.float32, device=DEVICE)
            else:
                action = torch.tensor(action, dtype=torch.float32, device=DEVICE)
            
            if action.dim() == 1:
                action = action.unsqueeze(1)
            
            next_encoded_state, reward = self.dynamics_net(encoded_state, action) 
            return next_encoded_state, reward
           
           
           
    def prediction (self , encoded_state) :
               policy , value = self.prediction_net(encoded_state )
               return policy , value


class MLPRepresentation(nn.Module) :
    def __init__ (self , input_size , output):
               super().__init__()
               self.fc1 = nn.Linear(input_size , 256)
               self.fc2 = nn.Linear(256 , output)
    def forward (self , x) :
               x = F.relu(self.fc1(x))
               x = self.fc2(x)
               x = (x - x.min() ) / (x.max() - x.min() + 1e-5)
               return x

class DynamicsNetwork (nn.Module) :
    def __init__ (self , state_size , action_size , out_size):
               super().__init__()
               self.fc1 = nn.Linear(state_size +1 , 256)
            #    self.fc2 = nn.Linear(256 , out_size)
               self.fc_state = nn.Linear(256 , out_size)
               self.fc_reward = nn.Linear(256, 1)
    def forward (self , state ,action) :
               x = torch.cat((state , action ) , dim = 1 )
               x = F.relu(self.fc1(x))
               nxt_state = self.fc_state(x)
               reward = self.fc_reward(x)
               return nxt_state , reward

class PredictionNetwork(nn.Module) :
    def __init__ (self , state , action , value) :
        super().__init__()
        self.fc1 = nn.Linear(state , 256 )
        self.fc_policy = nn.Linear(256 , action)
        self.fc_value = nn.Linear(256 , value)
    def forward (self , state )  :
        x= F.relu(self.fc1(state))
        plclogit = self.fc_policy(x)
        vallogits = self.fc_value(x)
        return plclogit , vallogits



