BASH_ENV=~/.bashrc
ROOT_PATH=/workspaces/bike-sharing-serving
PIPENV_PIPFILE=$ROOT_PATH/Pipfile
PYENV_VERSION=3.9.16

export PATH=$PATH:/home/codespace/.pyenv/shims
export PIPENV_PIPFILE=$PIPENV_PIPFILE
export PYENV_VERSION=$PYENV_VERSION
# pipenv run python $ROOT_PATH/batch_prediction.py >> $ROOT_PATH/cron.log 2>&1

PIPENV_PYTHON=/home/codespace/.local/share/virtualenvs/bike-sharing-serving-3MZy-oAL/bin/python
$PIPENV_PYTHON $ROOT_PATH/batch_prediction.py >> $ROOT_PATH/cron.log 2>&1
