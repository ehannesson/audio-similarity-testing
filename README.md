# Audio Similarity Testing
## Contribution Workflow
### Setup

1. Fork this repository by clicking the **Fork** button at the top right corner of this page (stop and read https://help.github.com/articles/about-forks/ if you don't know what a fork is).
You may name your fork whatever you want.
2. Clone the fork of your repository to your machine and add the original repository as a remote. If `username` is your GitHub username and `forkname` is the name of your new fork, do the following:
```bash
# Navigate to wherever you want to put your folder, then clone the fork.
cd ~/path/to/
git clone https://username@github.com/username/forkname.git

# Add the original repository as a remote.
cd forkname
git remote add upstream https://github.com/ehannesson/audio-similarity-testing.git
```
Within your repository folder, `origin` refers to your fork repository and `upstream` refers to this source repository.

![xkcd:git](https://imgs.xkcd.com/comics/git.png)

### Workflow

Git is all about careful coordination and communication.
You work on the code on your computer copy and make sure your online copy matches, then you use your online copy to submit requests for changes to the online repository.
In turn, you update your computer copy to make sure it has all new changes from the source repository.

##### Sync your Fork with the Source

Open command prompt (or git bash) and cd into your repository folder.
Run `git branch` to check your current branch.
If a star appears next to `develop`, you are on the default branch, called develop.
**NEVER MAKE EDITS WHILE ON DEVELOP;** keep it as a clean copy of the source repository.
Update `develop` with the following commands.
```bash
git checkout develop                    # Switch to the develop branch.
git pull upstream develop               # Get updates from the source repo.
git push origin develop                 # Push updates to your fork.
```
##### Make Edits

1. Create a new branch for editing.
```bash
git checkout develop                    # Switch to the develop branch.
git checkout -b newbranch               # Make a new branch and switch to it. Pick a good branch name.
```
**Only make new branches from the `develop` branch** (when you make a new brach with `git branch`, it "branches off" of the current branch).
To switch between branches, use `git checkout <branchname>`.

2. Make edits, saving your progress at reasonable segments.
```bash
git add filethatyouchanged
git commit -m "<a DESCRIPTIVE commit message>"
```
3. Push your working branch to your fork once you're done making edits.
```bash
git push origin newbranch               # Make sure the branch name matches your current branch
```
4. Create a pull request.
Go to the page for this repository.
Click the green **New Pull Request** button.

##### Clean Up

After your pull request is merged, you need to get those changes (and any other changes from other contributors) into your `develop` branch and delete your working branch.
If you continue to work on the same branch without deleting it, you are risking major merge conflicts.

1. Update the `develop` branch.
```bash
git checkout develop            # Switch to develop.
git pull upstream develop       # Pull changes from the source repo.
git push origin develop	        # Push updates to your fork.
```
2. Delete your working branch. **Always do this after (and only after) your pull request is merged.**
```bash
git checkout newbranch          # Switch to the branch where your now-merged edits came from.
git merge develop               # Reconcile the commits in newbranch and develop.
git checkout develop            # Switch back to develop.
git branch -d newbranch         # Delete the working branch.
git push origin :newbranch      # Tell your fork to delete the example branch.
```

See https://help.github.com/articles/creating-a-pull-request-from-a-fork/ for more info on pull requests.
GitHub's [git cheat sheet](https://services.github.com/on-demand/downloads/github-git-cheat-sheet.pdf) may also be helpful.

![xkcd:git commit](https://imgs.xkcd.com/comics/git_commit.png)
