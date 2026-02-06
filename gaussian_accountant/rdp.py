# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

from .accountant import IAccountant
from .analysis import rdp as privacy_analysis


class RDPAccountant(IAccountant):
    DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(
        range(12, 200)
    )  # 100太少

    def __init__(self):
        super().__init__()
        self.history = []  # history of noise multiplier, sample rate, and steps

    def step(self, *, noise_multiplier: float, sample_rate: float, num_steps: int = 1):
        """
        Record privacy steps.

        Args:
            noise_multiplier: Noise multiplier (sigma) used in this step
            sample_rate: Sampling rate (batch_size / dataset_size)
            num_steps: Number of steps with these parameters (default: 1)
        """
        if len(self.history) >= 1:
            last_noise_multiplier, last_sample_rate, last_num_steps = self.history.pop()
            if (
                last_noise_multiplier == noise_multiplier
                and last_sample_rate == sample_rate
            ):
                self.history.append(
                    (
                        last_noise_multiplier,
                        last_sample_rate,
                        last_num_steps + num_steps,
                    )
                )
            else:
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, last_num_steps)
                )
                self.history.append((noise_multiplier, sample_rate, num_steps))

        else:
            self.history.append((noise_multiplier, sample_rate, num_steps))

    def get_privacy_spent(
        self, *, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ) -> Tuple[float, float]:
        if not self.history:
            return 0, 0

        if alphas is None:
            alphas = self.DEFAULT_ALPHAS
        rdp = sum(
            [
                privacy_analysis.compute_rdp(
                    q=sample_rate,
                    noise_multiplier=noise_multiplier,
                    steps=num_steps,
                    orders=alphas,
                )
                for (noise_multiplier, sample_rate, num_steps) in self.history
            ]
        )
        eps, best_alpha = privacy_analysis.get_privacy_spent(
            orders=alphas, rdp=rdp, delta=delta
        )
        return float(eps), float(best_alpha)

    def get_epsilon(
        self, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ):
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
            alphas: List of RDP orders (alphas) used to search for the optimal conversion
                between RDP and (epd, delta)-DP
        """
        eps, _ = self.get_privacy_spent(delta=delta, alphas=alphas)
        return eps

    def __len__(self):
        return len(self.history)

    @classmethod
    def mechanism(cls) -> str:
        return "rdp"
