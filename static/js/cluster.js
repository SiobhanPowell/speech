      var data = {{ output | safe }}

//        self.colours = {'Residential L1': '#bf812d', 'Residential L2': '#dfc27d', 'MUD L2': '#f6e8c3', 'Workplace L2': '#80cdc1', 'Public L2': '#01665e', 'Public DCFC': '#003c30'}
//       const color = ["lightgreen", "lightblue", "red", "orange", "yellow"];
      const color = ['#bf812d', '#dfc27d', '#f6e8c3', '#80cdc1', '#01665e', '#003c30'];

      // Create SVG and padding for the chart
      const svg = d3
        .select("#chart")
        .append("svg")
        .attr("height", 400)
        .attr("width", 600);

      const strokeWidth = 1.5;
      const margin = { top: 0, bottom: 60, left: 60, right: 20 };
      const chart = svg.append("g").attr("transform", `translate(${margin.left},0)`);

      const width = +svg.attr("width") - margin.left - margin.right - strokeWidth * 2;
      const height = +svg.attr("height") - margin.top - margin.bottom;
      const grp = chart
        .append("g")
        .attr("transform", `translate(-${margin.left - strokeWidth},-${margin.top})`);

      // Create stack
      const stack = d3.stack().keys(["line1", "line2", "line3", "line4", "line5", "line6"]);
      const stackedValues = stack(data);
      const stackedData = [];
      
      // Copy the stack offsets back into the data.
      stackedValues.forEach((layer, index) => {
         const currentStack = [];
         layer.forEach((d, i) => {
            currentStack.push({
               values: d,
               x: data[i].x
            });
         });
         stackedData.push(currentStack);
       });

      // Create scales
      const yScale = d3
        .scaleLinear()
        .range([height, 0])
        .domain([0, d3.max(stackedValues[stackedValues.length - 1], dp => dp[1])]);
     
 
      const xScale = d3
         .scaleLinear()
         .range([0, width])
         .nice(d3.timeFormat("%H"))
         .domain(d3.extent(data, dataPoint => dataPoint.x));

      const area = d3
         .area()
         .x(dataPoint => xScale(dataPoint.x))
         .y0(dataPoint => yScale(dataPoint.values[0]))
         .y1(dataPoint => yScale(dataPoint.values[1]));

      const series = grp
         .selectAll(".series")
         .data(stackedData)
         .enter()
         .append("g")
         .attr("class", "series");

      series
         .append("path")
         .attr("transform", `translate(${margin.left},0)`)
         .style("fill", (d, i) => color[i])
         .attr("stroke", "steelblue")
         .attr("stroke-linejoin", "round")
         .attr("stroke-linecap", "round")
         .attr("stroke-width", strokeWidth)
         .attr("d", d => area(d));

      // Add the X Axis 
      xchart = chart
         .append("g")
         .attr("transform", `translate(0,${height})`)
         .call(d3.axisBottom(xScale).ticks("4"));

      // Add the Y Axis 
      ychart = chart
         .append("g")
         .attr("transform", `translate(0, 0)`)
         .call(d3.axisLeft(yScale));
         
      // X Axis Label
      xchart
         .append("text")
         .text("Hour of Day")
         .attr("class", "axis-label")
         .attr("x","220")
         .attr("y","40")

     // Y Axis Label 
      ychart
         .append("text")
         .text("MW")
         .attr("class", "axis-label")
         .attr("x","-140")
         .attr("y","-50")
         .attr("transform", "rotate(-90)")
      
