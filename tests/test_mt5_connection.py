import os
import time
import MetaTrader5 as mt5
import dotenv

dotenv.load_dotenv()

def main():
    terminal_path = os.getenv("MT5_TERMINAL_PATH")

    print("📌 Testing MT5 connection")
    print("Terminal path from env:", terminal_path)

    # تلاش اول برای initialize
    if terminal_path:
        ok = mt5.initialize(path=terminal_path, timeout=60000)
    else:
        ok = mt5.initialize(timeout=60000)

    print("Initialize result:", ok, "last_error:", mt5.last_error())

    if not ok:
        print("❌ Initialization failed, trying portable=True...")
        mt5.shutdown()
        time.sleep(0.5)
        if terminal_path:
            ok = mt5.initialize(path=terminal_path, timeout=60000, portable=True)
        else:
            ok = mt5.initialize(timeout=60000, portable=True)
        print("Re-initialize result:", ok, "last_error:", mt5.last_error())

    if not ok:
        print("❌ Could not initialize MT5, aborting.")
        return

    # کمی صبر تا IPC آماده شود
    for _ in range(25):
        if mt5.terminal_info() is not None:
            break
        time.sleep(0.2)

    ti = mt5.terminal_info()
    print("Terminal info:", ti)

    ai = mt5.account_info()
    print("Account info:", ai)

    if ai is None:
        print("⚠️ Not logged in. Trying to login using env variables...")
        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        if login and password and server:
            success = mt5.login(int(login), password=password, server=server)
            print("Login result:", success, "last_error:", mt5.last_error())
            print("Account info (after login):", mt5.account_info())
        else:
            print("❌ Missing MT5_LOGIN / MT5_PASSWORD / MT5_SERVER in environment")

    mt5.shutdown()
    print("✅ Test finished")

if __name__ == "__main__":
    main()
