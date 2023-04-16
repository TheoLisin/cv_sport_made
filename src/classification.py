from torch import nn
from torch import Tensor


class ClassificationHead(nn.Module):
    def __init__(self, num_of_classes: int, start_size: int = 1280, depth: int = 4) -> None:
        if start_size // depth < 2 * num_of_classes:
            raise ValueError("Too big depth.")

        self.num_of_classes = num_of_classes
        self.start_size = start_size
        self.depth = depth
        super().__init__()

        self.head = self._build_head()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)
    
    def _build_head(self) -> nn.Module:
        head = []
        step = self.start_size // (self.depth + 1)
        input_size = self.start_size
        
        for _ in range(1, self.depth):
            output_size = input_size - step
            head.append(self._make_part(input_size, output_size))
            head.append(nn.ReLU())
            input_size = output_size

        head.append(
            nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(input_size, self.num_of_classes),
            )
        )

        return nn.Sequential(*head)

    def _make_part(self, input_size: int, output_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_size, output_size, bias=True),
            nn.BatchNorm1d(output_size)
        )