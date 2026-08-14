import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = "/Users/chenyuan/Documents/develop/paper-assistant/outputs/019ff624-360b-7702-91d9-40ca910d9fce/paper_agent_failed_files.xlsx";
const input = await FileBlob.load(sourcePath);
const workbook = await SpreadsheetFile.importXlsx(input);

const sheets = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 4000 });
const detail = workbook.worksheets.getItem("失败明细").getRange("A1:D57").values;
const summary = workbook.worksheets.getItem("错误汇总").getRange("A1:B7").values;
const counts = {};
for (const row of detail.slice(1)) {
  const code = String(row[1] ?? "unknown");
  const reason = String(row[2] ?? "");
  const key = `${code} | ${reason}`;
  counts[key] = (counts[key] ?? 0) + 1;
}
process.stdout.write(`${sheets.ndjson}\n${JSON.stringify({ rows: detail.length - 1, counts, summary }, null, 2)}\n`);
