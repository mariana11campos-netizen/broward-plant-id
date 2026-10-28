# Putting this on GitHub and turning on the website

Your empty repo: https://github.com/Nick-48-85/broward-plant-id

## Option A: in the browser (no git needed)

1. Download and unzip `broward-plant-id.zip`.
2. Open https://github.com/Nick-48-85/broward-plant-id and click **uploading an existing file**.
3. Open the unzipped `broward-plant-id` folder, select **everything inside it**
   (including the `docs` folder), and drag it onto the page. GitHub keeps the folders.
   Hidden files (`.gitignore`, `docs/.nojekyll`) may not drag over; that's fine.
4. Click **Commit changes**.

## Option B: with git

```bash
cd broward-plant-id
git init -b main
git add .
git commit -m "Broward plant ID: AlexNet native vs invasive classifier and website"
git remote add origin https://github.com/Nick-48-85/broward-plant-id.git
git push -u origin main
```

## Turn on GitHub Pages

1. In the repo, go to **Settings > Pages**.
2. Under **Build and deployment**, Source: **Deploy from a branch**.
3. Branch: **main**, folder: **/docs**. Click **Save**.
4. After a minute or two the site is live at
   https://nick-48-85.github.io/broward-plant-id/

To show training results on the site, run `python train.py --site` and upload
the new `docs/results.json`.
