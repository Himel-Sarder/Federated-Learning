## 1) Open the project folder

Example:

```powershell
cd E:\FederatedL
```

Make sure you can see:

* `server.py`
* `client.py`
* `utils.py` (if you use it)

---

## 2) Start the server (Terminal 1)

Open a new terminal and run:

```powershell
python server.py
```

You should see something like:

* gRPC server running
* rounds starting

Keep this terminal running.

---

## 3) Start clients (Terminal 2 and Terminal 3)

Open another terminal (Terminal 2):

```powershell
python client.py
```

Open another terminal (Terminal 3):

```powershell
python client.py
```

Now the server will detect 2 clients and start FL rounds.

---

## 4) If you want 3 clients

Change server strategy to require 3 clients, then run client 3 times:

**server.py (important lines):**

```python
min_fit_clients=3,
min_available_clients=3,
min_evaluate_clients=3,
```

Then run in three different terminals:

```powershell
python client.py
python client.py
python client.py
```

---

## 5) Common problems and fixes

### Problem: Client cannot connect

* Start server first
* Check address is same in both files:

  * server: `"127.0.0.1:8080"`
  * client: `"127.0.0.1:8080"`

### Problem: Port already in use

Change port to 8081 in both server and client:

* server:

  ```python
  server_address="127.0.0.1:8081"
  ```
* client:

  ```python
  server_address="127.0.0.1:8081"
  ```
