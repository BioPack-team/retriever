//find and download and parse
const { ONTOLOGIES } = require("./config");
const { parseFile } = require("./parser");
const child_process = require("child_process");

//iterate over ontologies obj
for (let ontology of Object.values(ONTOLOGIES)) {
  const url = ontology.url;
  console.log(`Downloading ${url}`);
  //use curl to download the file
  let download_process = child_process.exec(`curl -o ../data/${ontology.name}.owl -L ${url}`);
  download_process.stdout.pipe(process.stdout);
  download_process.stderr.pipe(process.stderr);

  download_process.on("close", code => {
    if (code !== 0) {
      console.log(`Error downloading ${url}`);
      return;
    }
    console.log(`Done downloading ${url} with code ${code}`);

    // use robot to convert owl to json
    child_process.execSync(
      `../robot convert -i ../data/${ontology.name}.owl --format json -o ../data/${ontology.name}.json`,
    );
    console.log(`Parsing ${url}`);
    parseFile(`../data/${ontology.name}.json`);
    console.log(`Cleaning up ${url}`);
    child_process.exec(`rm ../data/${ontology.name}.owl`);
    child_process.exec(`rm ../data/${ontology.name}.json`);
  });
}
