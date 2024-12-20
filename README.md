# generic assistant

Use it to run an AI app.


# Template for python container for AI

This repo extend https://github.com/NeuronAddict/python-template (python template).

It add to this template : 

- notebook facilities
- nbclean with pre-commit to add book
-  

## Doc for the initial template

When you build a python container with a requirement.txt without explicit versions, it sometimes fail (too often!).

Why? This is because without versions in this file, your build use the last versions of all dependencies.
This work at a moment (when develop) but can fail when YOU build code.

To avoid that, this template help us to provides a unique build for a commit.

- Docker rootless container for python
- Include only python dependencies

## How To

### set python dependencies

- Add needed dev dependencies in dev-requirements.in. There are not deployed on final container (like pytest).
- Add run dependencies (packages you need to run in production) in requirements.in. Add versions only if needed.

#### Exemple

requirements.in
```
django
psycopg2
```

### set dev env

```
$ python -m venv venv # may be python3
$ source venv/bin/activate
$ pip install pip-tools # only when dev-requirements.txt is void
$ # remove your local index-url and options to not leak this on github
$ pip-compile --no-emit-index-url --no-emit-options requirements.in
$ # add a constraint to dev-requirements to avoid a different version on dev env
$ pip-compile --constraint=requirements.txt --no-emit-index-url --no-emit-options dev-requirements.in 
$ pip install -r requirements.txt
$ pip install -r dev-requirements.txt
```

Commit all requirements files (in and txt). Like this your commit give a unique (and reproducible!) build.

Now all *.txt files has unique versions and your commit work now AND anytime in the future.

### update packages

To update or add packages, simply use pip-tools :

```
$ # remove your local index-url and options to not leak this on github
$ pip-compile --upgrade --no-emit-index-url --no-emit-options requirements.in
$ # add a constraint to dev-requirements to avoid a different version on dev env
$ pip-compile --constraint=requirements.txt --no-emit-index-url --no-emit-options dev-requirements.in 
$ # now *.txt files are updated with new versions
$ pip install -r dev-requirements.txt
$ pip install -r requirements.txt
$ git add *
$ git commit
```

https://pypi.org/project/pip-tools/

This create a new commit with all new versions. You can make a rollback if needed.

## Good practices

### Exclude files from docker container

Ignore useless files with a dockerignore to not include they in your container :
 
- Dockerfile # can leak info
- .git (can be very big and leak a lot of info)
- *.md # useless
- venv/ (may be big!)
- doc (should not be in a container)
- *.in
- dev-requirements.txt
- .gitlab-ci.yaml, Jenkinsfile (can leak info)
- .dockerignore itself to not leak ignored containers

### One commit, one build

Create a new container version when upgrade packages to keep versions consistent accross time.
A commit should not give a different build when rebuild after a long time.

### Use staged build

Like the example, use a staged build to :

- Add all required packages on build container
- Add only required packages on production build and clean apt-get cache in same layer

https://docs.docker.com/build/building/multi-stage/
