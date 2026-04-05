
## QUICK SETUP (Windows/Mac/Linux)

### 1\. INSTALL UV (one time)

Windows PowerShell:
```
irm https://astral.sh/uv/install.ps1 | iex
```


Mac / Linux / Git Bash:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```


Then reopen your terminal and check:
```
uv --version
```
### 2\. CLONE THE REPOSITORY

```
git clone https://github.com/martavise/Seminar_RLDMUU_AI_agents_contracts.git
```
```
cd Seminar_RLDMUU_AI_agents_contracts.git
```

### 3\. CREATE THE ENVIRONMENT (from lock file)


```
uv sync --frozen
```

Sync or update environment (use frozen to match exact versions):
```
uv sync --frozen
```

### ADD A NEW LIBRARY

```
uv add <library-name>

uv lock

git add pyproject.toml uv.lock

git commit -m "Add <library-name> dependency"

git push
```