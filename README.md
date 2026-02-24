# ETHGlobal Bangkok 2024: Agent + Meme Coin Modeling Prototype

Hackathon prototype combining crypto market data collection, a sequence-modeling experiment, and an on-chain agent workflow (Coinbase AgentKit / CDP).

## Demo

- X demo post (Bangkok hackathon): https://x.com/BarruecoEnrique/status/1858012841523163469

## What Is In This Repo

This repository contains two main tracks explored during the hackathon:

1. **Market data / modeling experiments**
- dataset collection for Solana meme coins
- sequence preparation + LSTM training (`crypto_price_prediction.py`)
- model outputs/checkpoints under `model_runs/`

2. **Agent workflow prototype**
- `my_bot.py` uses LangChain/LangGraph + Coinbase CDP AgentKit tools
- interactive/autonomous modes for on-chain actions and market exploration

## Repository Layout

- `crypto_price_prediction.py` - LSTM training pipeline on OHLCV sequences
- `get solana meme training dataset .ipynb` - dataset recreation notebook
- `my_bot.py` - CDP AgentKit-based agent prototype
- `solana_meme_coins/` - split CSV dataset parts
- `solana_meme_coins.csv`, `base_coins*.csv` - prepared data snapshots
- `model_runs/` - generated model run artifacts
- `sketch.ipynb` - hackathon exploration notebook

## Setup

1. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment variables for the agent workflow (when using `my_bot.py`)
- `OPENAI_API_KEY`
- `GRAPH_API_KEY`
- `GRAPH_SUBGRAPH_ID`
- `CDP_API_KEY_NAME`
- `CDP_API_KEY_PRIVATE_KEY`

3. Recreate the Solana dataset (optional)
- Run `get solana meme training dataset .ipynb`

4. Train the sequence model

```bash
python crypto_price_prediction.py
```

5. Run the agent prototype (optional)

```bash
python my_bot.py
```

## Notes

- This is a hackathon codebase and includes raw artifacts / local files.
- Some files (wallet snapshots, `.env`, model outputs) should be treated as local development artifacts.
- A productionized version would separate data ingestion, training, and agent execution into smaller modules.
