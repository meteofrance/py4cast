## Installation on Météo-France HPC platform
The instructions here follow Météo-France's security policy, and are subjected to change when the related policy is updated.
Installing on HPC is typically made with conda, following the README.md instructions (README.md#Install with conda).
However, there are two configuration settings to add :

- Set the content of your `$HOME/.condarc` file (create it if needed) to

```
ssl_verify: false
channels:
  - pytorch
  - nvidia
  - pyg
  - conda-forge
channel_priority: strict
```

 - Set the content of your `$HOME/.config/pip/pip.conf` file (create it if needed) to :

```
[global]
timeout = 3600
cert = /opt/softs/certificats/proxy1.pem
```

 - Make sure that you have none of these environment variables set :

```
REQUEST_CA_BUNDLE
CURL_CA_BUNDLE
SSL_CERT_FILE
```

