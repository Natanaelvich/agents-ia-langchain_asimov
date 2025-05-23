import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import { Logger } from "../Logger";

/**
 * Define types for mock data
 */
interface DataFrameRow {
  PassengerId: number;
  Survived: number;
  Pclass: number;
  Name: string;
  Sex: string;
  Age: number | null;
  SibSp: number;
  Parch: number;
  Ticket: string;
  Fare: number;
  Cabin: string | null;
  Embarked: string | null;
}

// Mock data
const mockData: DataFrameRow[] = [
  {
    PassengerId: 1,
    Survived: 1,
    Pclass: 3,
    Name: "Braund, Mr. Owen Harris",
    Sex: "male",
    Age: 22,
    SibSp: 1,
    Parch: 0,
    Ticket: "A/5 21171",
    Fare: 7.25,
    Cabin: null,
    Embarked: "S"
  },
  // ... add more mock data as needed
];

/**
 * DataFrame Tools
 */
export const dfTools: DynamicStructuredTool[] = [
  new DynamicStructuredTool({
    name: "df_shape",
    description: "Get the shape of the DataFrame",
    schema: z.object({}),
    func: async () => {
      return JSON.stringify([mockData.length, Object.keys(mockData[0]).length]);
    },
  }),
  new DynamicStructuredTool({
    name: "df_head",
    description: "Get the first n rows of the DataFrame",
    schema: z.object({
      n: z.number().describe("Number of rows to return"),
    }),
    func: async ({ n }) => {
      const result = mockData.slice(0, n).map((row: DataFrameRow) => {
        const newRow: Record<string, any> = {};
        Object.keys(row).forEach(key => {
          newRow[key] = row[key as keyof DataFrameRow];
        });
        return newRow;
      });
      return JSON.stringify(result);
    },
  }),
  new DynamicStructuredTool({
    name: "df_describe",
    description: "Get statistical description of the DataFrame",
    schema: z.object({
      column: z.string().describe("Column name to describe"),
    }),
    func: async ({ column }) => {
      if (!mockData[0] || !(column in mockData[0])) {
        throw new Error(`Column ${column} not found in DataFrame`);
      }
      const values = mockData
        .map((row: DataFrameRow) => row[column as keyof DataFrameRow])
        .filter((val): val is number => typeof val === 'number');
      const stats = {
        count: values.length,
        mean: values.reduce((a, b) => a + b, 0) / values.length,
        min: Math.min(...values),
        max: Math.max(...values),
      };
      return JSON.stringify(stats);
    },
  }),
]; 