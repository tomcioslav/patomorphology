# Sync training run folders between this machine and the remote GPU box.
#
# Runs are gitignored (only runs/README is tracked), so they have to move
# over SSH. rsync here is resumable: --partial keeps half-transferred files
# and the retry loop reconnects after a dropped connection, picking up where
# it left off. None of these targets ever delete — syncs are purely additive
# merges, so pulling can't clobber a local run and pushing can't clobber a
# remote one.
#
# Requires a modern rsync on both ends (--append-verify needs rsync 3.x).
# On macOS that means the Homebrew rsync, not Apple's bundled /usr/bin/rsync.
#
# Override the remote with:  make pull-runs REMOTE=user@host REMOTE_ROOT=/path/to/repo

REMOTE      ?= tomek@ubuntu
REMOTE_ROOT ?= /home/tomek/Projects/Kraftcode/patomorphology
LOCAL_ROOT  := $(CURDIR)

SSH_OPTS := ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=5 -o TCPKeepAlive=yes
RSYNC    := rsync -avh --progress --partial --append-verify --timeout=60 -e '$(SSH_OPTS)'

# rsync wrapped in a reconnect loop: $(1) = source, $(2) = destination.
define rsync_retry
until $(RSYNC) $(1) $(2); do echo ">>> rsync dropped, retrying in 10s..."; sleep 10; done
endef

.PHONY: help pull-runs push-runs pull-run push-run list-remote-runs

help:
	@echo "Run-folder sync targets:"
	@echo "  make pull-runs                  pull ALL remote runs into ./runs/"
	@echo "  make push-runs                  push ALL local runs to the remote"
	@echo "  make pull-run RUN=<folder>      pull one run from the remote"
	@echo "  make push-run RUN=<folder>      push one run to the remote"
	@echo "  make list-remote-runs           list run folders on the remote"
	@echo ""
	@echo "Remote: $(REMOTE):$(REMOTE_ROOT)/runs/"

## Pull every remote run folder into local runs/ (additive merge).
pull-runs:
	$(call rsync_retry,$(REMOTE):$(REMOTE_ROOT)/runs/,$(LOCAL_ROOT)/runs/)

## Push every local run folder to the remote runs/ (additive merge).
push-runs:
	$(call rsync_retry,$(LOCAL_ROOT)/runs/,$(REMOTE):$(REMOTE_ROOT)/runs/)

## Pull a single run: make pull-run RUN=sam_finetune-sam_deep-...
pull-run:
	@test -n "$(RUN)" || { echo "usage: make pull-run RUN=<folder-name>"; exit 1; }
	$(call rsync_retry,$(REMOTE):$(REMOTE_ROOT)/runs/$(RUN),$(LOCAL_ROOT)/runs/)

## Push a single run: make push-run RUN=unet-nmsc-2x-...
push-run:
	@test -n "$(RUN)" || { echo "usage: make push-run RUN=<folder-name>"; exit 1; }
	$(call rsync_retry,$(LOCAL_ROOT)/runs/$(RUN),$(REMOTE):$(REMOTE_ROOT)/runs/)

## List run folders on the remote (newest last).
list-remote-runs:
	@ssh $(REMOTE) 'ls -1 $(REMOTE_ROOT)/runs/'
