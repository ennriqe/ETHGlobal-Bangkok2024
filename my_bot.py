import os
import sys
import time

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import CDP Agentkit Langchain Extension.
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API keys from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GRAPH_API_KEY = os.getenv('GRAPH_API_KEY')
GRAPH_SUBGRAPH_ID = os.getenv('GRAPH_SUBGRAPH_ID')
CDP_API_KEY_NAME = os.getenv('CDP_API_KEY_NAME')
CDP_API_KEY_PRIVATE_KEY = os.getenv('CDP_API_KEY_PRIVATE_KEY')

# Check if keys are present
if not all([OPENAI_API_KEY, GRAPH_API_KEY, GRAPH_SUBGRAPH_ID, CDP_API_KEY_NAME, CDP_API_KEY_PRIVATE_KEY]):
    print("Warning: One or more required API keys not found in environment variables!")
    print("Current environment variables:", dict(os.environ))

# Configure a file to persist the agent's CDP MPC Wallet Data.
# wallet_data_file = f"wallet_data_{int(time.time())}_{os.urandom(4).hex()}.txt"
wallet_data_file = "wallet_data_1731771477_72b8d0990.txt"
# Initialize CDP Agentkit wrapper


# Create toolkit from wrapper


def initialize_agent(state_modifier):
    """Initialize the agent with CDP Agentkit."""
    # Initialize LLM.
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)  

    
    wallet_data = None
    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()
            print(f"Loaded wallet data from {wallet_data_file}")
    

    # Configure CDP Agentkit Langchain Extension.
    values = {
        "cdp_api_key_name": CDP_API_KEY_NAME,
        "cdp_api_key_private_key": CDP_API_KEY_PRIVATE_KEY,
        "cdp_wallet_data": wallet_data,
        "network_id": 'base-mainnet'
    }

    agentkit = CdpAgentkitWrapper(**values)

    # persist the agent's CDP MPC Wallet Data.

    wallet_data = agentkit.export_wallet()
    # with open(wallet_data_file, "w") as f:
    #     f.write(wallet_data)
    # with open(walletwha

    # Initialize CDP Agentkit Toolkit and get tools.
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()
    tools = [tool for tool in tools if tool.name != 'wow_buy_token']

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Chatbot Example!"}}

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=state_modifier,
    ), config


# Autonomous Mode
def run_autonomous_mode(agent_executor, config, interval=10):
    """Run the agent autonomously with specified intervals."""
    print("Starting autonomous mode...")
    while True:
        try:
            # Provide instructions autonomously
            thought = (
                "Be creative and do something interesting on the blockchain. "
                "Choose an action or set of actions and execute it that highlights your abilities."
            )

            # Run agent in autonomous mode
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=thought)]}, config
            ):
                if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                print("-------------------")

            # Wait before the next action
            time.sleep(interval)

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)


# Chat Mode
def run_chat_mode(agent_executor, config):
    """Run the agent interactively based on user input."""
    print("Starting chat mode... Type 'exit' to end.")
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() == "exit":
                break

            # Run agent with the user's input in chat mode
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}, config
            ):
                if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                print("-------------------")

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)


# Mode Selection
def choose_mode():
    """Choose whether to run in autonomous or chat mode based on user input."""
    while True:
        print("\nAvailable modes:")
        print("1. chat    - Interactive chat mode")
        print("2. auto    - Autonomous action mode")

        choice = input("\nChoose a mode (enter number or name): ").lower().strip()
        if choice in ["1", "chat"]:
            return "chat"
        elif choice in ["2", "auto"]:
            return "auto"
        print("Invalid choice. Please try again.")

def get_RAG_query(intent):
    intent = "Intent: "+intent
    prompt = (
        intent + """Produce a .query() to filter the dataframe called base_coins to get the intent of the user.

    This is how the df looks like:
    base_coins[['symbol', 'name', 'current_price', 
        'market_cap_rank', 'total_volume_mm', 'price_change_percentage_24h', 'ath',
        'ath_date', 'market_cap_mm']]

    Metadata of what each column means:
    'symbol': Trading symbol of the coin
    'name': Full name of the coin
    'current_price': Current trading price in USD
    'market_cap_rank': Rank by market capitalization
    'total_volume_mm': 24h trading volume in millions
    'price_change_percentage_24h': 24h price change percentage
    'ath': All-time high price
    'ath_date': Date of ATH
    'market_cap_mm': Market cap in millions
    

    Info of which columns represent which categories:

    # Boolean columns indicating token categories/features:
    'meme',                  # Whether token is a meme coin
    'base_ecosystem',        # Whether token is on Base chain
    'cat_themed',           # Whether token has cat theme
    'base_meme',            # Whether token is a Base chain meme
    'dog_themed',           # Whether token has dog theme  
    'farcaster_ecosystem',  # Whether token is Farcaster-related
    'frog_themed',          # Whether token has frog theme
    'the_boy's_club',       # Whether token is part of Boy's Club
    'ai_meme',              # Whether token is AI-themed meme
    'parody_meme',          # Whether token is a parody
    'layer_3_(l3)',         # Whether token is L3-related
    'politifi',             # Whether token is politics-themed
    'country_themed_meme',  # Whether token represents a country
    'gaming_(gamefi)',      # Whether token is gaming/GameFi related
    'nft',                  # Whether token has NFT features
    'on_chain_gaming'       # Whether token is for on-chain gaming

    Please return **only** the query string needed to perform the query. **Do not** include any quotes, code fences, or additional text.
    """
)


    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY, organization = "org-38tcB2xwAMriGXlXET77TRAu")
    response_format={"type": "text"}

    final_dict = {}


    messages = [
        {
            "role": "system",
            "content": """Your output will be placed inside of output this way: 
                filtered_coins = base_coins.query(output) 
                Therefore it must:
                1. Use proper pandas query syntax
                2. Use `&` instead of 'and'
                3. Wrap column names with spaces in backticks
                4. Be directly executable with no other text
                5. Use == True/False for boolean comparisons
                
                Example valid queries:
                "cat_themed == True & market_cap_mm < 100"
                "`total_volume_mm` > 1 & price_change_percentage_24h > 0"
                """
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    completion = client.chat.completions.create(
        response_format=response_format,

        # model="o1-preview",
        model= "gpt-4o",
        messages=messages,
    )
    text = completion.choices[0].message.content 
    return text

def get_coins():
    import time

from pycoingecko import CoinGeckoAPI
import pandas as pd

pd.options.display.max_columns = None

def get_base_meme_coins(dev_boolean):
    """
    Fetches coins from the 'Base Meme' category on CoinGecko and includes their contract addresses.
    
    Returns:
        pd.DataFrame: Dataframe containing coins data with contract addresses.
    """

    if dev_boolean:
        base_coins = pd.read_csv('base_coins_with_contracts.csv')
        base_coins['total_volume_mm'] = base_coins['total_volume']/1e6
        return base_coins
    else:
        pass

    # Step 1: Fetch all categories and find the ID for 'Base Meme'
    # Import timeout context manager
    from contextlib import contextmanager
    import signal

    @contextmanager
    def timeout(seconds):
        def signal_handler(signum, frame):
            raise TimeoutError("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

    cg = CoinGeckoAPI()

    # First read existing CSV if it exists
    existing_coins = pd.DataFrame()
    try:
        existing_coins = pd.read_csv('base_coins_with_contracts.csv')
        print(f"Found {len(existing_coins)} existing coins in CSV")
    except FileNotFoundError:
        print("No existing CSV found, will create new one")

    # Step 1: Fetch all categories and find the ID for 'Base Meme'
    print("Fetching categories from CoinGecko...")
    categories = cg.get_coins_categories_list()

    category_id = next((category['category_id'] for category in categories if category['name'].lower() == 'base meme'), None)

    if not category_id:
        print("Category 'Base Meme' not found.")

    print(f"Category 'Base Meme' found with ID: {category_id}")

    # Step 2: Fetch coins from the 'Base Meme' category
    print(f"Fetching coins in category '{category_id}'...")
    coins = cg.get_coins_markets(vs_currency='usd', category=category_id, per_page=250, page=1)

    if not coins:
        print("No coins found in the 'Base Meme' category.")

    # Step 3: Add contract addresses and categories to the coin data
    coins_with_contracts = []
    for coin in coins:
        # Skip if coin already exists in CSV
        if not existing_coins.empty and coin['symbol'] in existing_coins['symbol'].values:
            print(f"Skipping {coin['name']} - already in CSV")
            continue
            
        print(f"Querying details for coin: {coin['name']} (ID: {coin['id']})")
        
        # Try up to 3 times with 10 second timeout
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with timeout(10):
                    coin_details = cg.get_coin_by_id(coin['id'])
                    break
            except TimeoutError:
                if attempt < max_retries - 1:
                    print(f"Timeout querying {coin['name']}, retrying...")
                    continue
                else:
                    print(f"Failed to query {coin['name']} after {max_retries} attempts, skipping...")
                    continue
            except Exception as e:
                print(f"Error querying {coin['name']}: {str(e)}")
                continue
                
        contract_address = coin_details.get('platforms', {}).get('base', None)
        
        # Get categories for this coin
        categories = coin_details.get('categories', [])
        
        coin_data = {
            'symbol': coin['symbol'],
            'name': coin['name'],
            'current_price': coin['current_price'],
            'market_cap_rank': coin.get('market_cap_rank'),
            'total_volume': coin.get('total_volume'),
            'price_change_percentage_24h': coin.get('price_change_percentage_24h'),
            'ath': coin.get('ath'),
            'ath_date': coin.get('ath_date'),
            'market_cap_mm': coin.get('market_cap', 0) / 1_000_000 if coin.get('market_cap') else None,
            'contract_address': contract_address
        }
        
        # Add binary columns for each category
        for category in categories:
            # Convert category name to lowercase and replace spaces with underscores
            category_col = category.lower().replace(' ', '_').replace('-', '_')
            coin_data[category_col] = 1
            
        coins_with_contracts.append(coin_data)
        
        # Convert current coins to DataFrame and save after each coin
        current_coins = pd.DataFrame(coins_with_contracts)
        
        # Fill NaN values in category columns with 0
        category_cols = [col for col in current_coins.columns if col not in ['symbol', 'name', 'current_price', 
                                                                        'market_cap_rank', 'total_volume',
                                                                        'price_change_percentage_24h', 'ath',
                                                                        'ath_date', 'market_cap_mm', 'contract_address']]
        for col in category_cols:
            current_coins[col] = current_coins[col].fillna(0)

        # Combine with existing coins
        if not existing_coins.empty:
            # Add any missing category columns to existing_coins
            for col in category_cols:
                if col not in existing_coins.columns:
                    existing_coins[col] = 0
            base_coins = pd.concat([existing_coins, current_coins], ignore_index=True)
        else:
            base_coins = current_coins

        # Save to CSV after each coin
        base_coins['total_volume_mm'] = base_coins['total_volume']/1e6
        base_coins.to_csv('base_coins_with_contracts.csv', index=False)
        print(f"Updated CSV with {len(base_coins)} total coins")
        
        time.sleep(5)  # To respect API rate limits

    print("Finished processing all coins")
def get_candlestick_chart_data(TOKEN_CONTRACT_ADDRESS):
    import requests
    import pandas as pd
    import time
    from datetime import datetime, timedelta
    from tqdm import tqdm
    import os
    import numpy as np

    # ---------------------------
    # Configuration
    # ---------------------------
    # Subgraph ID (ensure it's correct)
    API_KEY = GRAPH_API_KEY
    SUBGRAPH_ID = GRAPH_SUBGRAPH_ID
    GRAPHQL_ENDPOINT = f'https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}'

    # Define the token contract address you want to analyze

    # TOKEN_CONTRACT_ADDRESS= '0x532f27101965dd16442e59d40670faf5ebb142e4'.lower() 
    # Define the USDC contract address (Ethereum Mainnet)
    WETH_CONTRACT_ADDRESS = '0x4200000000000000000000000000000000000006'.lower()

    # Define the number of intervals and interval duration
    NUM_INTERVALS = 30
    INTERVAL_MINUTES = 15
    INTERVAL_SECONDS = INTERVAL_MINUTES * 60  # 900 seconds

    # CSV Output File
    OUTPUT_CSV = 'token_price_history_usdc_pools.csv'

    # ---------------------------
    # Define GraphQL Queries
    # ---------------------------

    # Query to fetch the token details based on contract address
    token_query = """
    query GetToken($id: ID!) {
    token(id: $id) {
        id
        name
        symbol
        decimals
        lastPriceUSD
    }
    }
    """

    # Query to fetch Swap events for pools within a time range
    # Update the swaps query to include token amounts
    swaps_query = """
    query GetSwaps($pool_ids: [ID!], $startTime: BigInt!, $endTime: BigInt!, $first: Int!, $skip: Int!) {
    swaps(
        first: $first,
        skip: $skip,
        where: {
        pool_in: $pool_ids,
        timestamp_gte: $startTime,
        timestamp_lt: $endTime
        },
        orderBy: timestamp,
        orderDirection: asc
    ) {
        id
        timestamp
        amountInUSD
        amountOutUSD
        amountIn
        amountOut
        tokenIn {
        id
        symbol
        decimals
        }
        tokenOut {
        id
        symbol
        decimals
        }
    }
    }
    """

    # Query to fetch pools associated with a token and USDC
    pools_query = """
    query GetPools($token_id: ID!, $usdc_id: ID!) {
    liquidityPools(
        where: {
        inputTokens_contains: [$token_id, $usdc_id]
        }
    ) {
        id
        inputTokens {
        id
        }
    }
    }
    """

    # ---------------------------
    # Helper Functions
    # ---------------------------

    def fetch_graphql(query, variables):
        """
        Generic function to send a GraphQL query to The Graph's API.
        """
        try:
            response = requests.post(
                GRAPHQL_ENDPOINT,
                headers={"Content-Type": "application/json"},
                json={
                    "query": query,
                    "variables": variables
                }
            )
            if response.status_code == 200:
                result = response.json()
                if 'errors' in result:
                    print("GraphQL Errors:", result['errors'])
                    return None
                return result['data']
            else:
                print(f"Query failed with status code {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"Exception during GraphQL query: {e}")
            return None

    def get_token(token_address):
        """
        Fetch token details based on the contract address.
        """
        variables = {
            "id": token_address
        }
        data = fetch_graphql(token_query, variables)
        if data and 'token' in data and data['token']:
            return data['token']
        else:
            print(f"No token found with address {token_address}")
            return None

    def get_pools(token_id, usdc_id):
        """
        Fetch pool IDs associated with a given token ID and USDC.
        """
        variables = {
            "token_id": token_id,
            "usdc_id": usdc_id
        }
        data = fetch_graphql(pools_query, variables)
        if data and 'liquidityPools' in data:
            pools = data['liquidityPools']
            # Optional: Verify that each pool indeed includes USDC
            valid_pools = []
            for pool in pools:
                input_tokens = [token['id'] for token in pool['inputTokens']]
                if usdc_id in input_tokens:
                    valid_pools.append(pool['id'])
            
            return valid_pools
        return []

    def get_swaps(pool_ids, start_time, end_time):
        """
        Fetch all Swap events for given pool IDs within the time range.
        Handles pagination.
        """
        swaps = []
        first = 1000  # Maximum allowed by The Graph
        skip = 0
        while True:
            variables = {
                "pool_ids": pool_ids,
                "startTime": str(start_time),
                "endTime": str(end_time),
                "first": first,
                "skip": skip
            }
            data = fetch_graphql(swaps_query, variables)
            if data and 'swaps' in data:
                batch = data['swaps']
                if not batch:
                    break
                swaps.extend(batch)
                if len(batch) < first:
                    break
                skip += first
                time.sleep(0.2)  # To respect rate limits
            else:
                break
        return swaps



    def aggregate_swaps(swaps):
        """
        Calculate consistent price as BRETT/WETH across all swaps
        """
        if not swaps:
            return None
        
        prices = []
        volumes = []
        swaps_sorted = sorted(swaps, key=lambda x: int(x['timestamp']))

        
        for swap in swaps_sorted:
            try:
                if swap['tokenIn']['id'].lower() == TOKEN_CONTRACT_ADDRESS.lower():
                    # BRETT -> WETH
                    brett_amount = float(swap['amountIn']) / 1e18
                    weth_amount = float(swap['amountOut']) / 1e18
                elif swap['tokenOut']['id'].lower() == TOKEN_CONTRACT_ADDRESS.lower():
                    # WETH -> BRETT
                    brett_amount = float(swap['amountOut']) / 1e18
                    weth_amount = float(swap['amountIn']) / 1e18
                else:
                    continue
                    
                if weth_amount > 0:
                    # Calculate BRETT/WETH ratio
                    price = brett_amount / weth_amount
                    volumes.append(float(swap['amountInUSD']))
                    prices.append(price)
                    
            except (ValueError, ZeroDivisionError) as e:
                print(f"Error processing swap: {e}")
                continue
        
        if not prices:
            return None
            
        # Calculate volume-weighted metrics
        total_volume = sum(volumes)
        
        print(f"\nInterval Summary:")
        print(f"Number of swaps: {len(prices)}")
        print(f"Token/WETH ratio range: {min(prices):.6f} - {max(prices):.6f}")
        print(f"Total volume: ${total_volume:,.2f}")
        
        return {
            "open": prices[0],
            "close": prices[-1],
            "high": max(prices),
            "low": min(prices),
            "volume": total_volume
        }

    def save_to_csv(df, filename):
        """
        Save DataFrame to CSV. If file exists, append without headers.
        Else, write with headers.
        """
        try:
            if not os.path.isfile(filename):
                df.to_csv(filename, index=False)
            else:
                df.to_csv(filename, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error saving to CSV: {e}")

    # ---------------------------
    # Main Execution
    # ---------------------------

    # Fetch the token details
    print("Fetching token details...")
    token = get_token(TOKEN_CONTRACT_ADDRESS)
    if not token:
        print("Token not found. Exiting.")


    token_id = token['id']
    symbol = token['symbol']
    print(f"Processing token: {symbol} (ID: {token_id})")

    # Fetch pools associated with the token and USDC
    pools = get_pools(token_id, WETH_CONTRACT_ADDRESS)
    if not pools:
        print(f"No USDC pools found for token {symbol}. Exiting.")


    print(f"Found {len(pools)} USDC pools for token {symbol}.")

    # Define the time intervals (15 minutes each, last NUM_INTERVALS intervals)
    intervals = []
    current_time = int(time.time())
    for i in range(NUM_INTERVALS):
        end_time = current_time - i * INTERVAL_SECONDS
        start_time = end_time - INTERVAL_SECONDS
        intervals.append((start_time, end_time))
    # Reverse to have chronological order
    intervals = intervals[::-1]

    # Create empty list to store data
    data = []

    # Iterate over each 15-minute interval
    print(f"Fetching data for the last {NUM_INTERVALS} intervals of {INTERVAL_MINUTES} minutes each using USDC pools...")
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    data_lock = Lock()
    
    def process_interval(interval):
        interval_start, interval_end = interval
        # Fetch Swap events for the pool within the interval
        swaps = get_swaps(pools, interval_start, interval_end)
        aggregated = aggregate_swaps(swaps)
        
        interval_start_dt = datetime.utcfromtimestamp(interval_start).strftime('%Y-%m-%d %H:%M:%S')
        interval_end_dt = datetime.utcfromtimestamp(interval_end).strftime('%Y-%m-%d %H:%M:%S')
        
        if aggregated:
            row = {
                "symbol": symbol,
                "interval_start": interval_start_dt,
                "interval_end": interval_end_dt,
                "open": aggregated['open'],
                "close": aggregated['close'],
                "high": aggregated['high'],
                "low": aggregated['low'],
                "volume": aggregated['volume']
            }
        else:
            print(f"No swap data for interval {interval_start_dt} - {interval_end_dt}")
            row = {
                "symbol": symbol,
                "interval_start": interval_start_dt,
                "interval_end": interval_end_dt,
                "open": np.nan,
                "close": np.nan,
                "high": np.nan,
                "low": np.nan,
                "volume": 0.0
            }
        
        with data_lock:
            data.append(row)
        time.sleep(0.1)  # To respect rate limits
        
    # Use ThreadPoolExecutor to process intervals in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(process_interval, intervals), total=len(intervals), desc="Processing Intervals"))

    # Create DataFrame from collected data
    df = pd.DataFrame(data)
    print("\nData collection complete. Data stored in DataFrame 'df'")
    return df



def plot_candlestick_chart(candelstick_data, actual_prediction):
    import mplfinance as mpf
    import pandas as pd 
    import matplotlib.pyplot as plt
    # Convert interval_start to datetime index
    df = candelstick_data.copy()
    df['interval_start'] = pd.to_datetime(df['interval_start'])
    df.set_index('interval_start', inplace=True)

    # Create OHLCV DataFrame in the format required by mplfinance
    ohlcv_data = df[['open', 'high', 'low', 'close', 'volume']]

    # Create a new DataFrame for the prediction point
    next_timestamp = df.index[-1] + pd.Timedelta(minutes=15)  # Assuming 15 min intervals
    prediction_df = pd.DataFrame(index=[next_timestamp], 
                            data={'close': [actual_prediction]})

    # Create the plot
    fig, axes = mpf.plot(ohlcv_data,
                        type='candle',
                        title='TOKEN/WETH Price',
                        ylabel='Price Ratio',
                        volume=True,
                        style='yahoo',
                        figsize=(12, 8),
                        returnfig=True)

    # Add prediction point
    axes[0].scatter(len(df), actual_prediction, color='red', marker='*', s=200, 
                    label='Prediction')
    axes[0].legend(['Prediction'])

    plt.show()
    return fig


def predict_price(df):
    import tensorflow as tf
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    # Load the saved model
    model = tf.keras.models.load_model('model_runs/20241117_025559/best_model.keras')
    print('model loaded')

    # Prepare the data
    def prepare_prediction_data(df, sequence_length=30):
        """
        Prepare the data for prediction in the format the model expects
        """
        # Select and reorder OHLCV columns to match training data format
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        data = df[ohlcv_cols].values
        
        # Create scaler and normalize the data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
        
        # Create sequence
        # Take the most recent sequence_length points
        sequence = normalized_data[-sequence_length:]
        
        # Reshape for model input (batch_size, sequence_length, features)
        sequence = sequence.reshape(1, sequence_length, len(ohlcv_cols))
        
        return sequence, scaler

    # Prepare the data
    print(df.shape)
    print(df.head())
    sequence, scaler = prepare_prediction_data(df)
    print('data prepared')

    # Make prediction
    prediction = model.predict(sequence)
    print('prediction made')
    # Inverse transform the prediction
    # Create a dummy array with the same shape as original data
    dummy_array = np.zeros((1, 5))  # 5 features (OHLCV)
    dummy_array[0, 3] = prediction[0, 0]  # Put prediction in the Close price position
    actual_prediction = scaler.inverse_transform(dummy_array)[0, 3]

    print(f"\nPredicted next close price ratio: {actual_prediction:.6f}")

    # Calculate percentage change from last close
    last_close = df['close'].iloc[-1]
    price_change = ((actual_prediction - last_close) / last_close) * 100

    print(f"Last close price ratio: {last_close:.6f}")
    print(f"Predicted change: {price_change:.2f}%")
    chart = plot_candlestick_chart(df, actual_prediction)
    return price_change, chart

import matplotlib.pyplot as plt
def get_all_coin_predictions(contracts):
    results = {}
    
    for contract in contracts:
        # TOKEN_CONTRACT_ADDRESS = '0x6921b130d297cc43754afba22e5eac0fbf8db75b'.lower()
        try:
            TOKEN_CONTRACT_ADDRESS = contract.lower()
            df = get_candlestick_chart_data(TOKEN_CONTRACT_ADDRESS)
            price_change, chart = predict_price(df)
            results[TOKEN_CONTRACT_ADDRESS] = {
            'price_change': price_change,
            'chart': chart
            }
            print(contract, results)

        except Exception as e:
            print(f"Error processing contract {contract}: {e}")
            continue
    print('going back to main')
    return results



def main():
    """Start the chatbot agent."""

    state_modifier="""You are a helpful agent that runs a trading bot, peple ask you to either buy a token or to check what tokens to buy, the companion agent that will actually tell them what to buy and what are the options is on the server, you need to find out what the user is looking for, they might ask you to buy an animal memecoin or a dex token, they might just ask you which memecoins are looking good or might pump soon, when in doubt ask more info to the user to narrow down the search but assume that just passing it over will work so dont bother the user too much. You will pass a prompt to the other agent defining wha the user wants to do, this other server agent will filter for the coins the user might be interest and give you a prediction of what to buy, if the user agrees you will actually buy it. 
    When you have a clear idea of what the user wants include #PROMPT TO THE SERVER# in your response, followed by the prompt you will send. If no clear buying intent is shown narrow it down until it is clear. This will be used to filter the coins the user might be interested in using RAG, therefore ONLY include this keyword in this part of the process and not in any point where you are actually trying to make the swap.
    Dont ever include #PROMPT TO THE SERVER# when executing the swap.
    If the user tells you to buy a token, you will buy it for the user, no need to ask the server, just execute the swap when you have decided on token and amount.
    """
    agent_executor, config = initialize_agent(state_modifier)

    while True:
        try:
            input_question = output_token_to_buy_info
            del output_token_to_buy_info
        except:
            input_question = input("What do you want to do?\n")
    

        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=input_question)]},
            {"configurable": {"thread_id": "my_first_agent"}}
        ):
            # Check if this is an agent message
            if "agent" in chunk:
                content = chunk["agent"]["messages"][0].content
                print(content)
                
                # Check for prompt to server
                if "#PROMPT TO THE SERVER#" in content:
                    
                    
                    try:
                        # Remove any quotes that might be in the query string
                        prompt_to_server = content.split("#PROMPT TO THE SERVER#")[1]
                        # print(prompt_to_server)
                        query = get_RAG_query(prompt_to_server)
                        print("Query:", query)  # Debug print
                        coins = get_base_meme_coins(dev_boolean=True)
                        print("Coins shape before query:", coins.shape)  # Debug print
                        query = query.strip("'\"")
                        filtered_coins = coins.query(query)
                        print("Filtered coins shape:", filtered_coins.shape)
                    except Exception as e:
                        print(f"Error executing query: {e}")
                        print("Available columns:", coins.columns.tolist())
                        pass

                    predictions = get_all_coin_predictions(filtered_coins['contract_address'])
                    print(predictions)
                    for contract, result in predictions.items():
                        direction = "up" if result['price_change'] > 0 else "down"
                        print(f"Token: {contract} predicted to go {direction} {result['price_change']:.2f}%")
                        result['chart'].show()
                    
                    user_input = input("What token would you like to buy:")
                    output_token_to_buy_info = "I would like to buy the token with crontract address " + user_input

            # Check if this is a tools message
            elif "tools" in chunk:
                print(chunk["tools"]["messages"][0].content)
            
            print("-------------------")

if __name__ == "__main__":
    print("Starting Agent...")
    main()
