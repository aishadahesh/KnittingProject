# Connecting to the UR5e robot

## Network layout

```
University LAN ── Ethernet ──┐
                             PC ── Wi-Fi ── TP-Link router ── LAN cable ── UR5e controller
                                            192.168.0.1                    192.168.0.101 (static)
```

| Device | Address | How it gets it |
|---|---|---|
| TP-Link router | `192.168.0.1` | Router itself; runs DHCP for `192.168.0.0/24` |
| UR5e controller | `192.168.0.101` / `255.255.255.0` | Static, set on the teach pendant |
| PC (Wi-Fi) | `192.168.0.100` (typically) | DHCP from the router |
| PC (Ethernet) | `132.74.121.x` | University DHCP; carries internet |

The PC can stay on both networks at once. Windows routes `192.168.0.x` out the Wi-Fi
(on-link route) and everything else out the Ethernet (lower default-route metric), so
the university internet keeps working while connected to the robot.

Wi-Fi networks: `TP-Link_CD91` (2.4 GHz) and `TP-Link_CD91_5G` (5 GHz). The 2.4 GHz
network has been more reliable (see *Problems* below).

## Robot side (teach pendant)

1. **Settings → System → Network**: Static address, IP `192.168.0.101`, mask
   `255.255.255.0`, gateway `192.168.0.1`.
2. **Remote control** enabled, and the pendant switched to **Remote** mode (top-right).
   Without Remote, the dashboard server refuses commands such as `play`/`power on`.
3. Robot powered on and brakes released (robot mode `RUNNING`).

## App side

The address lives in `config/params.json` as `ur5_robot_ip` (default `192.168.0.101`).

Ports the app and tools use:

| Port | Service |
|---|---|
| 29999 | Dashboard server (mode, safety status, play/stop) |
| 30001 | Primary interface (URScript) |
| 30002 / 30003 | Secondary / real-time interfaces |
| 30004 | RTDE (motion and state streaming in *Direct RTDE* transport) |

Only one RTDE control session is allowed: TeleopApp must not run while the app uses
*Direct RTDE*.

## Checking the connection

PowerShell, from the PC:

```powershell
# 1. Wi-Fi has a 192.168.0.x address (169.254.x.x means DHCP failed)
Get-NetIPAddress -InterfaceAlias Wi-Fi -AddressFamily IPv4

# 2. Router and robot answer
ping 192.168.0.1
ping 192.168.0.101

# 3. Robot ports are open
foreach ($p in 29999,30001,30002,30003,30004) {
  $c = New-Object Net.Sockets.TcpClient
  "$p : $($c.ConnectAsync('192.168.0.101',$p).Wait(2000))"; $c.Close()
}

# 4. Controller state through the dashboard server
$c = New-Object Net.Sockets.TcpClient('192.168.0.101',29999); $s=$c.GetStream()
$r=New-Object IO.StreamReader($s); $w=New-Object IO.StreamWriter($s); $w.AutoFlush=$true
$r.ReadLine()
foreach($cmd in 'is in remote control','robotmode','safetystatus'){ $w.WriteLine($cmd); "$cmd -> $($r.ReadLine())" }
$c.Close()
```

Healthy result: all pings reply, all five ports `True`, and
`is in remote control -> true`, `Robotmode: RUNNING`, `Safetystatus: NORMAL`.

## Problems seen and fixes

### Router stops answering while Wi-Fi still says "connected"
**Symptoms:** pings to both `192.168.0.1` and `.101` time out for seconds to minutes;
`netsh wlan show interfaces` shows `connected` with a strong signal; after a reconnect
the Wi-Fi gets `169.254.x.x` and `ipconfig /renew` fails with *unable to contact your
DHCP server*.
**Cause:** the TP-Link router hangs (seen on the 5 GHz network). Not a PC setting.
**Fix:** power-cycle the router, then reconnect the Wi-Fi so the PC gets a fresh lease:
```powershell
netsh wlan disconnect interface="Wi-Fi"
netsh wlan connect name="TP-Link_CD91" interface="Wi-Fi"
```
To stop Windows rejoining the 5 GHz network automatically:
```powershell
netsh wlan set profileparameter name="TP-Link_CD91_5G" connectionmode=manual
```
If it keeps happening, suspect the router's power supply or firmware, or use a cable
(USB-Ethernet adapter to the router) instead of Wi-Fi.

### PC still fails after the router is back
**Cause:** Windows kept the `169.254.x.x` fallback address from the outage.
**Fix:** reconnect the Wi-Fi (commands above). A static Wi-Fi address avoids this
entirely (admin PowerShell; no gateway, so internet stays on Ethernet):
```powershell
New-NetIPAddress -InterfaceAlias Wi-Fi -IPAddress 192.168.0.50 -PrefixLength 24
Set-NetIPInterface -InterfaceAlias Wi-Fi -Dhcp Disabled
```
Undo with `Set-NetIPInterface -InterfaceAlias Wi-Fi -Dhcp Enabled` and
`Remove-NetIPAddress -InterfaceAlias Wi-Fi -IPAddress 192.168.0.50`.

### Router answers but the robot drops out for a few seconds
**Possible causes:** loose LAN cable between router and controller, or an IP clash:
`.101` is likely inside the router's DHCP pool (TP-Link default `.100`–`.199`), so a
phone or laptop could be handed the robot's address.
**Check:** `Get-NetNeighbor -IPAddress 192.168.0.101`. The robot's MAC is
`00-30-d6-31-1b-3a`; any other MAC there means a clash.
**Fix:** reseat the cable; move the robot to an address outside the pool (e.g.
`192.168.0.20`) on the pendant and update `ur5_robot_ip` in `config/params.json`, or
reserve `.101` in the router's DHCP settings.

### Wi-Fi adapter power saving / Intel AX203 quirks
Short, regular stalls on the PC side can come from adapter power saving. Admin
PowerShell:
```powershell
Set-NetAdapterAdvancedProperty -Name Wi-Fi -DisplayName "802.11n/ac/ax Wireless Mode" -DisplayValue "3. 802.11ac"
Set-NetAdapterAdvancedProperty -Name Wi-Fi -DisplayName "MIMO Power Save Mode" -DisplayValue "No SMPS"
Set-NetAdapterAdvancedProperty -Name Wi-Fi -DisplayName "Packet Coalescing" -DisplayValue "Disabled"
Set-NetAdapterAdvancedProperty -Name Wi-Fi -DisplayName "Roaming Aggressiveness" -DisplayValue "1. Lowest"
# Wi-Fi power saving off on battery too
powercfg /setdcvalueindex SCHEME_CURRENT 19cbb8fa-5279-450e-9fac-8a3d5fedd0c1 12bbebe6-58d6-4636-95bb-3217ef867c1a 0
```

### Ports closed but ping works
- Controller still booting, or PolyScope not running: wait and retry.
- Port 30004 refused/busy: another RTDE client (TeleopApp, a stale app process) holds
  the session. Close it.
- Dashboard commands rejected: pendant is in **Local** mode; switch to **Remote**.

### RTDE drops mid-scan
The controller disconnects an RTDE client that falls behind reading the state stream.
The app streams at 125 Hz (`RTDE_FREQUENCY_HZ` in `src/ur5_robot.py`) for this reason;
heavy CPU work on the PC during a scan can still starve it. A weak or stalling Wi-Fi
link has the same effect. Prefer 2.4 GHz with good signal, or a cable.
