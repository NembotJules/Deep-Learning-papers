from torch import nn

class AlexNet(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride = 4),
            nn.ReLU(),
            nn.MaxPool2d(3, stride = 2),

            nn.Conv2d(96, 256, 5, stride = 1, padding = 2),
            nn.ReLU(), 
            nn.MaxPool2d(3, stride = 2),

            nn.Conv2d(256, 384, 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, stride = 1, padding = 1),
            nn.ReLU(),

            nn.Conv2d(384, 256, 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride = 2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9216, 4096)
        self.dropout = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(4096, 200)


    def forward(self, x): 
        x = self.feature_extraction(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x