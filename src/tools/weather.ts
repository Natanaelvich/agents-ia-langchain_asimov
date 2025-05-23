import { tool } from "@langchain/core/tools";
import { z } from "zod";
import axios from "axios";
import { Logger } from "../Logger";

/**
 * Tool Schemas
 */
const TemperatureArgsSchema = z.object({
  latitude: z.number().describe("Latitude of the location to get temperature"),
  longitude: z.number().describe("Longitude of the location to get temperature"),
});

type TemperatureArgs = z.infer<typeof TemperatureArgsSchema>;

/**
 * Tool to get current temperature
 */
export const getCurrentTemperature = tool(
  async (input: TemperatureArgs): Promise<string> => {
    try {
      const URL = "https://api.open-meteo.com/v1/forecast";
      const params = {
        latitude: input.latitude,
        longitude: input.longitude,
        hourly: "temperature_2m",
        forecast_days: 1,
      };

      const response = await axios.get(URL, { params });
      
      if (response.status === 200) {
        const result = response.data;
        const now = new Date();
        const hours = result.hourly.time.map((timeStr: string) => new Date(timeStr));
        
        // Find closest hour to current time
        const closestHourIndex = hours.reduce((closest: number, hour: Date, index: number) => {
          const currentDiff = Math.abs(hour.getTime() - now.getTime());
          const closestDiff = Math.abs(hours[closest].getTime() - now.getTime());
          return currentDiff < closestDiff ? index : closest;
        }, 0);

        return `${result.hourly.temperature_2m[closestHourIndex]}°C`;
      } else {
        throw new Error(`Request to API ${URL} failed: ${response.status}`);
      }
    } catch (error) {
      Logger.error(`Error getting temperature: ${error}`);
      throw error;
    }
  },
  {
    name: "getCurrentTemperature",
    description: "Gets the current temperature for given coordinates",
    schema: TemperatureArgsSchema,
  }
); 