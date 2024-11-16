1. **Create a Virtual Environment and Install Requirements**
   - Set up a Python virtual environment (`venv`).
   - Install the required dependencies using the `requirements.txt` file:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows, use venv\Scripts\activate
     pip install -r requirements.txt
     ```

2. **Obtain API Credentials**
   - Follow the [Coinbase AgentKit Quickstart Guide](https://docs.cdp.coinbase.com/agentkit/docs/quickstart) to acquire the necessary API credentials for `my_bot.py`.

3. **Recreate the Solana Dataset**
   - Use the `get solana meme training dataset.ipynb` notebook to regenerate the Solana dataset.

4. **Train the Model**
   - Run `crypto_price_prediction.py` to train and retain the model.

These steps will ensure everything works as intended.
