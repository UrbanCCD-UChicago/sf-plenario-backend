var TableChartHelper = {};
TableChartHelper.create = function(data, properties) {
  
  return new Highcharts.Chart({
      chart: {
          renderTo: 'chart_' + properties.name + '_' + properties.type,
          type: 'line'
      },
      title: {
          text: ''
      },
      legend: {
        enabled: false
      },
      subtitle: {
        //text: 'Source: ' + source
        enabled: false
      },
      xAxis: {
          title: {
              text: 'Date'
          },
          type: 'datetime'
      },
      yAxis: {
          title: {
              text: 'Value'
          }
      },
      tooltip: {
          formatter: function() {
              return TableChartHelper.toolTipDateFormat(
                  properties.time_agg, this.x) +': <b>'+ this.y + '</b>';
          }
      },
      plotOptions: {
          series: {
            marker: {
              radius: 0,
              states: {
                hover: {
                  enabled: true,
                  radius: 5
                }
              }
            },
            shadow: false,
            states: {
               hover: {
                  lineWidth: 3
               }
            }
          }
      },
      series: [{
          color: TableChartHelper.colors[properties.iteration],
          step: properties.duration == 'interval',
          name: properties.name,
          data: data
      }]
  });
}

TableChartHelper.toolTipDateFormat = function(interval, x) {
  if (interval == "year" || interval == "decade")
    return Highcharts.dateFormat("%Y", x);
  if (interval == "quarter")
    return Highcharts.dateFormat("%B %Y", x);
  if (interval == "month")
    return Highcharts.dateFormat("%B %Y", x);
  if (interval == "week")
    return Highcharts.dateFormat("%b %e, %Y", x);
  if (interval == "day")
    return Highcharts.dateFormat("%b %e, %Y", x);
  if (interval == "hour")
    return Highcharts.dateFormat("%H:00", x);
  else
    return 1;
}

TableChartHelper.colors = ["#A6761D", "#7570B3", "#D95F02", "#66A61E", "#E7298A", "#E6AB02", "#1B9E77"];
