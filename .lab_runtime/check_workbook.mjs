import fs from 'node:fs/promises';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const path = 'submission_2A202602586/experiments.xlsx';
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
workbook.recalculate();
for (const [sheetName, range] of [['Seeds', 'A1:D10'], ['Summary', 'A1:H14']]) {
  const check = await workbook.inspect({kind: 'table', range: `${sheetName}!${range}`, include: 'values,formulas', tableMaxRows: 14, tableMaxCols: 8, maxChars: 5000});
  console.log(check.ndjson);
  const preview = await workbook.render({sheetName, range, scale: 1.5, format: 'png'});
  await fs.writeFile(`.lab_runtime/${sheetName}.png`, new Uint8Array(await preview.arrayBuffer()));
}
console.log((await workbook.inspect({kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!', options: {useRegex:true,maxResults:20}, maxChars:2000})).ndjson);
