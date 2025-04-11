# darwin_fiftyone

Provides an integration between Voxel51 and V7 Darwin. This enables Voxel51 users to send subsets of their datasets to Darwin for annotation and review. The annotated data can then be imported back into Voxel51.

This integration is currently in beta.

## Install
1. Install the library
``` bash
pip install darwin-fiftyone 
```

2. Configure voxel51 to use it
```bash
cat ~/.fiftyone/annotation_config.json
```

```json
{
  "backends": {
    "darwin": {
      "config_cls": "darwin_fiftyone.DarwinBackendConfig",
      "api_key": "d8mLUXQ.**********************"
    }
  }
}
```
**Note**: Replace the api_key placeholder with a valid API key generated from Darwin.

## Development setup

1. Install the library and development dependencies
```bash
# Install all dependencies including development tools
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

2. Configure voxel51 to use it
```bash
cat ~/.fiftyone/annotation_config.json
```

```json
{
  "backends": {
    "darwin": {
      "config_cls": "darwin_fiftyone.DarwinBackendConfig",
      "api_key": "d8mLUXQ.**********************"
    }
  }
}
```
**Note**: Replace the api_key placeholder with a valid API key generated from Darwin.

3. (Optional) Install the [V7 plugin](https://docs.voxel51.com/integrations/v7.html) in V51 to push/pull data to V7 directly from V51.
```bash
fiftyone plugins download \
    https://github.com/voxel51/fiftyone-plugins \
    --plugin-names @voxel51/annotation
```
## Example Usage

Several notebooks can be found [here](./notebooks/) 


## API

In addition to the standard arguments provided by `dataset.annotate()`, we also support:

- `backend=darwin`, Indicates that the Darwin backend is being used.
- `atts`, Specifies attribute subannotations to be added in the labelling job
- `dataset_slug`, Specifies the name of the dataset to use or create on Darwin.
- `external_storage`, Specifies the sluggified name of the Darwin external storage and indicates that all files should be treated as external storage



## Testing 
Set up your environment with FiftyOne and Darwin integration settings. To find your team slug check the [Darwin documentation on dataset identifiers](https://docs.v7labs.com/reference/datasetidentifier) which has a section called "Finding Team Slugs:"

You'll also need an [API Key](https://docs.v7labs.com/docs/use-the-darwin-python-library-to-manage-your-data)

```bash
export FIFTYONE_ANNOTATION_BACKENDS=*,darwin
export FIFTYONE_DARWIN_CONFIG_CLS=darwin_fiftyone.DarwinBackendConfig
export FIFTYONE_DARWIN_API_KEY=******.*********
export FIFTYONE_DARWIN_TEAM_SLUG=your-team-slug-here
```
NB. E2E tests run in the IRL env and on a specific team with a specific external storage configuration. See [env.example](env.example)

## Supported Annotation Types

The integration currently supports bounding boxes, polygons (closed polylines), keypoints, and tags (classification). It also supports attributes, text, instance ids, and properties subtypes.

Future development work will focus on the addition of annotation and subannotation types. Do reach out if you have suggestions.

## TODO
- Support for read only external data storage
- Support for mask and keypoint skeleton types
