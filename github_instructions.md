# Instructions for Saving Your Project to GitHub

## Step 1: Create a GitHub Account (if you don't have one)
Visit [GitHub](https://github.com/) and sign up for an account if you don't already have one.

## Step 2: Install Git (if not already installed)
Check if Git is installed on your system:
```bash
git --version
```

If not installed, install Git:
```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install git

# For CentOS/RHEL
sudo yum install git

# For macOS (using Homebrew)
brew install git

# For Windows
# Download the installer from https://git-scm.com/download/win
```

## Step 3: Configure Git
Set up your Git identity:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 4: Create a New Repository on GitHub
1. Go to [GitHub](https://github.com/)
2. Click on the '+' icon in the upper right corner and select 'New repository'
3. Enter a repository name (e.g., "quantum-entanglement-characterization")
4. Add a description (optional)
5. Choose whether to make the repository public or private
6. Skip the initialization step (do not add README, .gitignore, or license for now)
7. Click "Create repository"

## Step 5: Initialize Git in Your Local Project
Navigate to your project directory:
```bash
cd /home/rithvik/Documents/ThesisWork/Work1/mlebay
```

Initialize Git:
```bash
git init
```

## Step 6: Create a .gitignore File
Create a .gitignore file to avoid uploading unnecessary files:
```bash
nano .gitignore
```

Add these entries:
````
# Ignore saved models
saved_models/
````

## Step 7: Add Your Files to the Repository
Add all your project files to the repository:
```bash
git add .
```

## Step 8: Commit Your Changes
Commit your changes with a message:
```bash
git commit -m "Initial commit"
```

## Step 9: Create a New Branch
Create a new branch for your changes:
```bash
git branch -M main
```

## Step 10: Add the Remote Repository
Add the remote repository URL:
```bash
git remote add origin https://github.com/your-username/quantum-entanglement-characterization.git
```

## Step 11: Push Your Changes
Push your changes to the remote repository:
```bash
git push -u origin main
```
