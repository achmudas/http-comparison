# Setting up

* Install Caddy https://caddyserver.com/docs/install
```
brew install caddy
```

### Misc

For Mac (ARM chip) I needed to point to newer curl version (the one which comes preinstalled doesn't support http 3)
* 
``` brew install curl
```

* Add `export PATH="$(brew --prefix curl)/bin:$PATH"` to ~/.zshrc (|#TODO add automatic instruction)
