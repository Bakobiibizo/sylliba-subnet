import os
from dotenv import load_dotenv

load_dotenv()


def miner_config():
    return {
        "logging": {
            "logging_dir": "~/.bittensor/miners",
        },
        "wallet": {
            "name": os.getenv("BT_WALLET_COLDKEY", "default"),
            "hotkey": os.getenv("BT_WALLET_HOTKEY", "default"),
            "path": os.getenv("BT_WALLET_PATH", "~/.bittensor/wallets/"),
        },
        "neuron": {
            "name": "default",
            "full_path": "~/mining/bittensor/bittensor-subnet-template/neurons/miner.py",
            "events_retention_size": 2 * 1024 * 1024 * 1024,
            "epoch_length": 10,
            "device": "cuda:0",
            "dont_save_events": False,
        },
        "subtensor": {
            "network": "local",
            "chain_endpoint": "http://localhost:9946",
            "mock": False,
        },
        "wandb": {
            "off": True,
            "offline": False,
            "notes": "",
            "project_name": "",
            "entity": "",
        },
        "blacklist": {
            "force_validator_permit": True,
            "allow_non_registered": False,
            "netuid": 1,
            "mock": False,
        },
    }


def validator_config():
    return {
        "logging": {
            "logging_dir": "~/.bittensor/validators",
        },
        "wallet": {
            "name": "bt_validator_test",
            "hotkey": "subnet_vali",
            "path": "~/.bittensor/wallets/",
        },
        "neuron": {
            "name": "default",
            "timeout": 10,
            "num_concurrent_forwards": 10,
            "sample_size": 50,
            "disable_set_weights": False,
            "moving_average_alpha": 0.1,
            "axon_off": False,
            "vpermit_tao_limit": 4096,
            "full_path": "~/mining/bittensor/bittensor-subnet-template/neurons/validator.py",
            "events_retention_size": 2 * 1024 * 1024 * 1024,
            "epoch_length": 10,
            "device": "cuda:0",
            "dont_save_events": False,
        },
        "subtensor": {
            "network": "local",
            "chain_endpoint": "http://localhost:9946",
            "mock": False,
        },
        "wandb": {
            "off": True,
            "offline": False,
            "notes": "",
            "project_name": "",
            "entity": "",
        },
        "blacklist": {
            "force_validator_permit": True,
            "allow_non_registered": False,
            "netuid": 1,
            "mock": False,
        },
    }
