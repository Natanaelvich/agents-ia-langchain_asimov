/**
 * Shared utility for routing agent responses
 * 
 * This module provides a common implementation for handling different types of agent responses:
 * 1. AIMessage with tool calls
 * 2. AIMessage with direct content
 * 3. Final response from the model
 * 4. Tool call in old format
 */

import { DynamicStructuredTool } from "@langchain/core/tools";
import { Logger } from "../Logger";

/**
 * Routes the response from the model to handle different types of responses
 * @param response The response from the model
 * @param tools The available tools for the agent
 * @returns The processed response as a string
 */
export async function routeResponse(response: any, tools: DynamicStructuredTool[]): Promise<string> {
  // Case 1: AIMessage with tool calls
  if (response.tool_calls && response.tool_calls.length > 0) {
    Logger.info(`Processing tool call from AIMessage\n`);
    const toolCall = response.tool_calls[0];
    const toolToRun = tools.find((tool) => tool.name === toolCall.name);
    
    if (!toolToRun) {
      throw new Error(`Tool ${toolCall.name} not found. Available tools: ${tools.map(t => t.name).join(', ')}`);
    }
    
    try {
      Logger.debug(`Executing tool ${toolCall.name} with input: ${JSON.stringify(toolCall.args)}\n`);
      const result = await (toolToRun as any)._call(toolCall.args);
      
      // Format the result based on the tool type
      if (toolCall.name === "getCurrentTemperature") {
        return `A temperatura atual é ${result}`;
      } else if (toolCall.name === "searchWikipedia") {
        return `Informações encontradas:\n${result}`;
      }
      
      return String(result);
    } catch (error) {
      throw new Error(`Error executing tool ${toolCall.name}: ${error}`);
    }
  }
  
  // Case 2: AIMessage with direct content
  else if (response.content) {
    Logger.info("Processing AIMessage response\n");
    return response.content;
  }
  
  // Case 3: Final response from the model
  else if (response.returnValues) {
    Logger.info("Processing final response from model\n");
    return response.returnValues.output;
  } 
  
  // Case 4: Tool call in old format
  else if (response.tool && response.toolInput) {
    Logger.info(`Processing tool call for ${response.tool}\n`);
    const toolToRun = tools.find((tool) => tool.name === response.tool);
    if (!toolToRun) {
      throw new Error(`Tool ${response.tool} not found. Available tools: ${tools.map(t => t.name).join(', ')}`);
    }
    try {
      Logger.debug(`Executing tool ${response.tool} with input: ${JSON.stringify(response.toolInput)}\n`);
      const result = await (toolToRun as any)._call(response.toolInput);
      
      // Format the result based on the tool type
      if (response.tool === "getCurrentTemperature") {
        return `A temperatura atual é ${result}`;
      } else if (response.tool === "searchWikipedia") {
        return `Informações encontradas:\n${result}`;
      }
      
      return String(result);
    } catch (error) {
      throw new Error(`Error executing tool ${response.tool}: ${error}`);
    }
  }

  // Case 5: Unexpected response format
  throw new Error(`Unexpected response format: ${JSON.stringify(response)}`);
} 