# Enhanced Bayesian Entanglement Estimation

## Git Setup and Push Instructions

1. Initialize git repository (if not already done):
```bash
git init
```

2. Configure git (if not already done):
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

3. Create .gitignore file:
```bash
echo "__pycache__/" > .gitignore
echo "plots/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "*.pyo" >> .gitignore
echo "*.pyd" >> .gitignore
echo ".pytest_cache/" >> .gitignore
echo ".coverage" >> .gitignore
echo "venv/" >> .gitignore
```

4. Stage your changes:
```bash
# Stage all files
git add .

# Or stage specific files
git add entchar.py plot.py requirements.txt
```

5. Commit your changes:
```bash
git commit -m "Enhanced Bayesian estimation implementation with plotting improvements"
```

6. Add remote repository (if not already done):
```bash
git remote add origin <your-repository-url>
```

7. Push to remote repository:
```bash
# First time push
git push -u origin main

# Subsequent pushes
git push
```

## Troubleshooting

If you encounter conflicts:
```bash
# Pull latest changes
git pull origin main

# Resolve conflicts, then
git add .
git commit -m "Resolved conflicts"
git push
```

To check repository status:
```bash
git status
```

To view commit history:
```bash
git log --oneline
```
