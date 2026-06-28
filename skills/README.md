# Orbit Wars CUDA PPO — 精華 Skills 文件

從 Distributed Orbit Wars 專案提取的可重複使用技術模式。這些 skills 同時存在於 `~/.claude/skills/` 作為 Claude Code 全域 skills，會在所有專案中自動生效。

## Skills 目錄

| Skill | 類型 | 說明 |
|-------|------|------|
| [pytorch-cuda-extension.md](pytorch-cuda-extension.md) | Reference | PyTorch CUDA 擴展編譯模式：CUDAExtension、nvcc flags、pybind11 綁定、零拷貝狀態 |
| [gpu-batched-env-simulation.md](gpu-batched-env-simulation.md) | Pattern | GPU 批量環境模擬架構：物理引擎+動作解碼+觀測編碼全在 GPU |
| [cuda-parity-testing.md](cuda-parity-testing.md) | Technique | CUDA 確定性 Parity 測試：fixture 狀態、binary dump、snapshot 比較 |

## 這些 Skills 來自同一個核心專案

Orbit Wars CUDA PPO 是一個將整個 RL 環境模擬器（物理引擎 + 動作解碼 + 觀測編碼）全部手寫成 CUDA C++ 的專案。核心檔案 `cpp/orbit_cuda.cu` 長達 2057 行，實現了：

- **物理引擎**：行星公轉、艦隊飛行、掃掠式碰撞偵測、太陽/邊界死亡、彗星生命週期、戰鬥結算
- **動作解碼器**：從 NN 輸出 `[launch, target, ship_bin]` 到驗證飛行軌跡（角度、ETA、碰撞檢查）
- **觀測編碼器**：16 步未來模擬 (forecast ledger) 為每位玩家產生 `[B, 17, 12]` 維度的預測特徵
- **大規模平行化**：4096 環境 × 4 玩家 = 16384 平行計算單元

## 如何在 Claude Code 中使用

這些 skills 已安裝在 `~/.claude/skills/`，Claude Code 會自動在相關任務時載入。觸發條件包括：

- 詢問 PyTorch CUDA extension / CUDAExtension / nvcc flags → 載入 `pytorch-cuda-extension`
- 設計 GPU 加速 RL 環境 / 批量環境模擬 → 載入 `gpu-batched-env-simulation`
- 撰寫 CUDA kernel 測試 / parity testing → 載入 `cuda-parity-testing`

## 授權

從 Orbit Wars CUDA PPO 專案提取，供所有專案重複使用。
