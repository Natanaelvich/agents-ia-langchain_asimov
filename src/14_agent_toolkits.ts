/**
 * Agent Toolkits
 * 
 * This example demonstrates how to use agent toolkits in LangChain:
 * 1. Pandas DataFrame Toolkit - For data analysis and manipulation
 * 2. SQL Database Toolkit - For database operations and queries
 */

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { config } from "dotenv";
import { Logger } from "./Logger";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { AgentAction, AgentFinish } from "@langchain/core/agents";
import { RunnablePassthrough } from "@langchain/core/runnables";
import { AIMessage, FunctionMessage, HumanMessage } from "@langchain/core/messages";
import axios from "axios";

// Load environment variables
config();

/**
 * Pandas DataFrame Toolkit
 * 
 * This toolkit provides tools for working with pandas DataFrames.
 * In a real implementation, you would use a proper DataFrame library.
 * For this example, we'll simulate DataFrame operations.
 */

// Define types for our mock data
type DataFrameData = {
  [key: string]: (number | string | null)[];
};

type MockTable = {
  columns: string[];
  sample: Record<string, any>[];
};

type MockDatabase = {
  [key: string]: MockTable;
};

// Mock DataFrame data
const mockData: DataFrameData = {
  PassengerId: [1, 2, 3, 4, 5],
  Survived: [0, 1, 1, 1, 0],
  Pclass: [3, 1, 3, 1, 3],
  Name: [
    "Braund, Mr. Owen Harris",
    "Cumings, Mrs. John Bradley",
    "Heikkinen, Miss Laina",
    "Futrelle, Mrs. Jacques Heath",
    "Allen, Mr. William Henry"
  ],
  Sex: ["male", "female", "female", "female", "male"],
  Age: [22, 38, 26, 35, 35],
  SibSp: [1, 1, 0, 1, 0],
  Parch: [0, 0, 0, 0, 0],
  Ticket: ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450"],
  Fare: [7.25, 71.28, 7.92, 53.10, 8.05],
  Cabin: [null, "C85", null, "C123", null],
  Embarked: ["S", "C", "S", "S", "S"]
};

// DataFrame Tools
const dfShapeTool = tool(
  async (): Promise<string> => {
    const rows = mockData.PassengerId.length;
    const cols = Object.keys(mockData).length;
    return `(${rows}, ${cols})`;
  },
  {
    name: "df_shape",
    description: "Returns the shape of the DataFrame (rows, columns)",
  }
);

const dfHeadTool = tool(
  async (input: { n: number }): Promise<string> => {
    const n = Math.min(input.n, mockData.PassengerId.length);
    const result: Record<string, any>[] = [];
    for (let i = 0; i < n; i++) {
      const row: Record<string, any> = {};
      for (const key in mockData) {
        row[key] = mockData[key][i];
      }
      result.push(row);
    }
    return JSON.stringify(result, null, 2);
  },
  {
    name: "df_head",
    description: "Returns the first n rows of the DataFrame",
    schema: z.object({
      n: z.number().describe("Number of rows to return"),
    }),
  }
);

const dfDescribeTool = tool(
  async (input: { column: string }): Promise<string> => {
    const column = input.column;
    if (!(column in mockData)) {
      throw new Error(`Column ${column} not found`);
    }

    const values = (mockData[column] as number[]).filter((v): v is number => typeof v === 'number');
    if (values.length === 0) {
      throw new Error(`Column ${column} does not contain numeric values`);
    }

    const sum = values.reduce((a: number, b: number) => a + b, 0);
    const mean = sum / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);

    return JSON.stringify({
      count: values.length,
      mean,
      min,
      max,
      sum
    }, null, 2);
  },
  {
    name: "df_describe",
    description: "Returns descriptive statistics for a numeric column",
    schema: z.object({
      column: z.string().describe("Name of the column to analyze"),
    }),
  }
);

const dfTools = [dfShapeTool, dfHeadTool, dfDescribeTool];

/**
 * SQL Database Toolkit
 * 
 * This toolkit provides tools for working with SQL databases.
 * For this example, we'll use a mock database with the Chinook schema.
 */

// Mock database schema
const mockTables: MockDatabase = {
  Album: {
    columns: ["AlbumId", "Title", "ArtistId"],
    sample: [
      { AlbumId: 1, Title: "For Those About To Rock We Salute You", ArtistId: 1 },
      { AlbumId: 2, Title: "Balls to the Wall", ArtistId: 2 },
      { AlbumId: 3, Title: "Restless and Wild", ArtistId: 2 }
    ]
  },
  Artist: {
    columns: ["ArtistId", "Name"],
    sample: [
      { ArtistId: 1, Name: "AC/DC" },
      { ArtistId: 2, Name: "Accept" },
      { ArtistId: 3, Name: "Aerosmith" }
    ]
  }
};

// SQL Tools
const listTablesTool = tool(
  async (): Promise<string> => {
    return Object.keys(mockTables).join(", ");
  },
  {
    name: "sql_list_tables",
    description: "Lists all tables in the database",
  }
);

const getTableSchemaTool = tool(
  async (input: { table: string }): Promise<string> => {
    const table = mockTables[input.table];
    if (!table) {
      throw new Error(`Table ${input.table} not found`);
    }
    return JSON.stringify({
      columns: table.columns,
      sample: table.sample
    }, null, 2);
  },
  {
    name: "sql_get_schema",
    description: "Returns the schema and sample data for a table",
    schema: z.object({
      table: z.string().describe("Name of the table"),
    }),
  }
);

const queryTool = tool(
  async (input: { query: string }): Promise<string> => {
    // In a real implementation, this would execute the SQL query
    // For this example, we'll return mock results
    if (input.query.toLowerCase().includes("select count")) {
      return "21"; // Mock result for album count
    }
    return JSON.stringify([
      { Name: "Iron Maiden", NumAlbums: 21 }
    ]);
  },
  {
    name: "sql_query",
    description: "Executes a SQL query and returns the results",
    schema: z.object({
      query: z.string().describe("SQL query to execute"),
    }),
  }
);

const sqlTools = [listTablesTool, getTableSchemaTool, queryTool];

/**
 * Agent Setup
 */
const chat = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-0125",
  temperature: 0,
});

/**
 * DataFrame Agent
 */
async function runDataFrameAgent(input: string) {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant that analyzes data using pandas DataFrame operations."],
    ["user", "{input}"],
    new MessagesPlaceholder("agent_scratchpad"),
  ]);

  const passThrough = RunnablePassthrough.assign({
    agent_scratchpad: (input: Record<string, unknown>) => {
      const steps = (input.intermediate_steps as [AgentAction, string][]) || [];
      return steps.map(([action, observation]) => {
        const message = new AIMessage({
          content: "",
          additional_kwargs: {
            function_call: {
              name: action.tool,
              arguments: JSON.stringify(action.toolInput),
            },
          },
        });
        return message;
      });
    },
  });

  const chain = RunnableSequence.from([
    passThrough,
    prompt,
    chat.bind({ tools: dfTools }),
  ]);

  let intermediateSteps: [AgentAction, string][] = [];

  while (true) {
    const response = await chain.invoke({
      input,
      intermediate_steps: intermediateSteps,
    });

    if (response.content && !response.tool_calls?.length) {
      return response.content;
    }

    if ('returnValues' in response && typeof response.returnValues === 'object' && response.returnValues !== null) {
      const returnValues = response.returnValues as { output: string };
      return returnValues.output;
    }

    const toolName = response.tool_calls?.[0]?.name;
    const toolArgs = response.tool_calls?.[0]?.args;

    if (!toolName || !toolArgs) {
      throw new Error(`Invalid tool call response: ${JSON.stringify(response)}`);
    }

    const action: AgentAction = {
      tool: toolName,
      toolInput: toolArgs,
      log: typeof response.content === 'string' ? response.content : ''
    };

    const toolToRun = dfTools.find((tool) => tool.name === toolName);
    if (!toolToRun) {
      throw new Error(`Tool ${toolName} not found`);
    }

    const result = await (toolToRun as any)._call(toolArgs);
    intermediateSteps.push([action, String(result)]);
  }
}

/**
 * SQL Agent
 */
async function runSQLAgent(input: string) {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant that analyzes data using SQL queries."],
    ["user", "{input}"],
    new MessagesPlaceholder("agent_scratchpad"),
  ]);

  const passThrough = RunnablePassthrough.assign({
    agent_scratchpad: (input: Record<string, unknown>) => {
      const steps = (input.intermediate_steps as [AgentAction, string][]) || [];
      return steps.map(([action, observation]) => {
        const message = new AIMessage({
          content: "",
          additional_kwargs: {
            function_call: {
              name: action.tool,
              arguments: JSON.stringify(action.toolInput),
            },
          },
        });
        return message;
      });
    },
  });

  const chain = RunnableSequence.from([
    passThrough,
    prompt,
    chat.bind({ tools: sqlTools }),
  ]);

  let intermediateSteps: [AgentAction, string][] = [];

  while (true) {
    const response = await chain.invoke({
      input,
      intermediate_steps: intermediateSteps,
    });

    if (response.content && !response.tool_calls?.length) {
      return response.content;
    }

    if ('returnValues' in response && typeof response.returnValues === 'object' && response.returnValues !== null) {
      const returnValues = response.returnValues as { output: string };
      return returnValues.output;
    }

    const toolName = response.tool_calls?.[0]?.name;
    const toolArgs = response.tool_calls?.[0]?.args;

    if (!toolName || !toolArgs) {
      throw new Error(`Invalid tool call response: ${JSON.stringify(response)}`);
    }

    const action: AgentAction = {
      tool: toolName,
      toolInput: toolArgs,
      log: typeof response.content === 'string' ? response.content : ''
    };

    const toolToRun = sqlTools.find((tool) => tool.name === toolName);
    if (!toolToRun) {
      throw new Error(`Tool ${toolName} not found`);
    }

    const result = await (toolToRun as any)._call(toolArgs);
    intermediateSteps.push([action, String(result)]);
  }
}

/**
 * Example Usage
 */
async function main() {
  try {
    Logger.info("Starting agent toolkits example\n");

    // Test DataFrame Agent
    Logger.info("Testing DataFrame Agent\n");
    const dfResponse = await runDataFrameAgent("What is the shape of the DataFrame?");
    Logger.success("DataFrame Agent response received\n");
    Logger.debug(`DataFrame Agent response: ${dfResponse}\n`);

    // Test SQL Agent
    Logger.info("Testing SQL Agent\n");
    const sqlResponse = await runSQLAgent("Which artist has the most albums?");
    Logger.success("SQL Agent response received\n");
    Logger.debug(`SQL Agent response: ${sqlResponse}\n`);

  } catch (error) {
    Logger.error(`Error in main: ${error}\n`);
  }
}

// Run the example
main(); 