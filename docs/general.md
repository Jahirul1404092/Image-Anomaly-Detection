
## Write Documents in MKDocs

One can write english and japanese to write document for `hamacho` product in MKDocs. Markdown language is used to write the content. Each language has its own `.md` file to write the content. For japanese document, `.jp` suffix should be added in the filename. For example 

- `dataset.md` (for english)
- `dataset.jp.md` (for japanese)

To create a document on particular cateory (i.e. dataset, model, training, inference, or API), the high-level structure would be as follows, under the `docs` folder.

```bash
docs # folder
├── <category> # folder
│   ├── [manage-{category}.md]
│   ├── [manage-{category}.jp.md]
│   ├── assets # folder
│   index.md
│   index.jp.md
```

For each category, there should be two files, one for english and one for japanese version. And any visual figure can be placed under the `assets` folder.

After the `.md or .jp.md` file is created, next, insert the file locations to the corresponding `index.md or index.jp.md` file. In that file, one can find a section called `Documentation`, where these `.md` filepath should be placed. For example:

```bash
## Documentation in index.md
- [category](./category/manage-category.md)
...
```

Lastly, update the order of `.md` files in the `nav` section of the `mkdocs.yml` file. It is required to keep categories sequences in the navigation panel of the site. Jus use `.md` files only, `jp` doc will be auto handled by `i18n` Plugin. For example:

```yaml
# Navigaiton panel
nav:
  - Home: index.md
  - General: general.md
  - API: ./api/manage-api.md
  - Config: ./config/manage-config.md
  - Docker: ./docker/manage-docker.md
  - Docker Compose easy start: ./docker/manage-docker-compose-webui.md
  - Dataset: ./dataset/manage-dataset.md
  - Model: ./model/manage-models.md
  - Training: ./training/manage-training.md
  - Inference: ./inference/manage-inference.md
```

- Ref. [PPT](https://chowagiken.sharepoint.com/:p:/g/CorporatePlanning/licence-business/EUtAiFY6vSdJjpYvFVV_dhcBt0XvXvCl670XkpFjZRyWOA?e=qhGKed).

## FAQ

#### What is `hamacho`?

A CUI product to perform visual anomaly detection, developed by Chowagiken.

#### How many models `hamacho` supports?

Currently it supports two type of models, i.e. `patchcore` and `padim`. Later, new models will be added.

#### Can we use `hamacho` even if we don't have bad image?

Yes, it can be trained only with good image.

